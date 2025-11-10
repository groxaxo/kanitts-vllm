"""VLLM-based text-to-speech generation logic with async streaming"""

import asyncio
import time
import torch
import numpy as np
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer

from config import (
    MODEL_NAME, START_OF_HUMAN, END_OF_TEXT, END_OF_HUMAN, END_OF_AI,
    TEMPERATURE, TOP_P, REPETITION_PENALTY, MAX_TOKENS, SAMPLE_RATE,
    BNB_QUANTIZATION
)


class VLLMTTSGenerator:
    def __init__(self, tensor_parallel_size=1, gpu_memory_utilization=0.9, max_model_len=2048, quantization=None, model_name=None, dtype="bfloat16"):
        """Initialize VLLM-based TTS generator with async streaming support

        Args:
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0 to 1.0)
            max_model_len: Maximum sequence length
            quantization: Quantization method (e.g., "bitsandbytes" for 4-bit quantization)
            model_name: Model name to load (defaults to MODEL_NAME from config)
            dtype: Model precision (e.g., "bfloat16", "float16")
        """
        # Use provided model_name or fall back to config
        self.model_name = model_name if model_name is not None else MODEL_NAME
        print(f"Loading VLLM AsyncLLMEngine model: {self.model_name}")
        
        # Use BnB quantization from config if not explicitly provided
        if quantization is None:
            quantization = BNB_QUANTIZATION
        
        if quantization:
            print(f"🔧 Using {quantization} quantization to reduce VRAM consumption")

        # Configure engine arguments
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=False,  # Allow CUDA graphs (reduces kernel launch overhead)
            max_num_seqs=1,  # Single sequence for TTS - enables better CUDA graph optimization
            dtype=dtype,  # Model precision (bfloat16 for RTX 30xx+)
            quantization=quantization,  # BitsAndBytes quantization for reduced VRAM
        )

        # Create async engine
        self.engine = None  # Will be initialized in async context
        self.engine_args = engine_args

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Pre-configure sampling parameters
        self.sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            repetition_penalty=REPETITION_PENALTY,
            stop_token_ids=[END_OF_AI],
        )

    async def initialize_engine(self):
        """Initialize the async engine - call this during startup to avoid lazy loading"""
        if self.engine is None:
            print("Initializing VLLM AsyncLLMEngine...")
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            print("VLLM AsyncLLMEngine initialized and ready!")

    def prepare_input(self, prompt):
        """Build custom input_ids with special tokens"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        # Add special tokens: [START_OF_HUMAN] + input_ids + [END_OF_TEXT, END_OF_HUMAN]
        start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
        end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

        # Convert to list for VLLM
        return modified_input_ids[0].tolist()

    async def _generate_async(self, prompt, audio_writer, max_tokens=MAX_TOKENS):
        """Async generator that streams tokens as they are generated

        Args:
            prompt: Text prompt to convert to speech
            audio_writer: StreamingAudioWriter instance to receive tokens
            max_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary with generation metrics and results
        """
        # Initialize engine if needed
        if self.engine is None:
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)

        # Prepare input_ids with special tokens
        input_ids = self.prepare_input(prompt)

        point_1 = time.time()

        # Override max_tokens if different from default
        if max_tokens != MAX_TOKENS:
            sampling_params = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=max_tokens,
                repetition_penalty=REPETITION_PENALTY,
                stop_token_ids=[END_OF_AI],
            )
        else:
            sampling_params = self.sampling_params

        # Generate unique request ID
        request_id = f"tts-{id(prompt)}-{time.time()}"

        # Stream tokens as they are generated
        all_token_ids = []
        audio_token_count = 0
        inside_speech = False

        # Add request to engine with TokensPrompt
        results_generator = self.engine.generate(
            {"prompt_token_ids": input_ids},
            sampling_params,
            request_id=request_id
        )

        async for request_output in results_generator:
            # Get newly generated tokens
            new_token_ids = request_output.outputs[0].token_ids

            # Find which tokens are new since last iteration
            num_new_tokens = len(new_token_ids) - len(all_token_ids)
            if num_new_tokens > 0:
                new_tokens = new_token_ids[-num_new_tokens:]
                all_token_ids.extend(new_tokens)

                # Stream each new token to audio_writer and count audio tokens
                for token_id in new_tokens:
                    # print(f"[VLLM] Token {len(all_token_ids)}: {token_id}")
                    audio_writer.add_token(token_id)

                    # Track audio tokens efficiently during streaming
                    if token_id == audio_writer.player.start_of_speech:
                        inside_speech = True
                    elif token_id == audio_writer.player.end_of_speech:
                        inside_speech = False
                    elif inside_speech:
                        audio_token_count += 1

        point_2 = time.time()
        generation_time = point_2 - point_1

        # Calculate Real Time Factor (RTF)
        # Audio codec runs at 12.5 fps, audio tokens come in groups of 4 per frame
        FRAMES_PER_SECOND = 12.5
        TOKENS_PER_FRAME = 4

        # Calculate audio duration: tokens / 4 = frames, frames / 12.5 = seconds
        num_frames = audio_token_count // TOKENS_PER_FRAME
        audio_duration = num_frames / FRAMES_PER_SECOND
        rtf = generation_time / audio_duration if audio_duration > 0 else 0

        # Calculate token counts
        prompt_tokens = len(input_ids)
        generated_tokens = len(all_token_ids)
        total_tokens = prompt_tokens + generated_tokens

        print(f"\n[VLLM] Generation complete. Prompt tokens: {prompt_tokens}, Generated tokens: {generated_tokens}, Total: {total_tokens}")
        print(f"       Audio tokens: {audio_token_count}, Frames: {num_frames}, Audio duration: {audio_duration:.2f}s")
        print(f"       Generation time: {generation_time:.2f}s, RTF: {rtf:.3f}")

        # OPTIMIZATION: Skip text decoding - it's slow and not needed for TTS

        return {
            'all_token_ids': all_token_ids,
            'generation_time': generation_time,
            'audio_duration': audio_duration,
            'rtf': rtf,
            'point_1': point_1,
            'point_2': point_2
        }

    def generate(self, prompt, audio_writer, max_tokens=MAX_TOKENS):
        """Generate speech tokens from text prompt with streaming

        This is a synchronous wrapper around the async streaming implementation.

        Args:
            prompt: Text prompt to convert to speech
            audio_writer: StreamingAudioWriter instance to receive tokens
            max_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary with generation metrics and results
        """
        # Try to get the current event loop, or create a new one if needed
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(self._generate_async(prompt, audio_writer, max_tokens))
        else:
            # Event loop is running, we need to run in a thread pool
            import concurrent.futures
            import threading

            result = None
            exception = None

            def run_in_new_loop():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(
                        self._generate_async(prompt, audio_writer, max_tokens)
                    )
                    new_loop.close()
                except Exception as e:
                    exception = e

            thread = threading.Thread(target=run_in_new_loop)
            thread.start()
            thread.join()

            if exception:
                raise exception

            return result

    async def generate_long_form_async(self, text, voice, player, max_chunk_duration=12.0,
                                       silence_duration=0.2, max_tokens=MAX_TOKENS):
        """Generate speech for long text by splitting into chunks with voice consistency

        This method handles texts longer than the model's training distribution (5-15s)
        by splitting into sentence-based chunks and generating each with the same voice.

        Args:
            text: Input text (can be any length)
            voice: Voice name for consistency (e.g., 'katie', 'alloy')
            player: LLMAudioPlayer instance for decoding audio
            max_chunk_duration: Target duration per chunk in seconds (default 12s)
            silence_duration: Duration of silence between chunks in seconds (default 0.2s)
            max_tokens: Maximum tokens per generation

        Returns:
            Dictionary with:
                - audio: Concatenated audio as numpy array
                - chunks_info: List of info dicts for each chunk
                - total_duration: Total audio duration in seconds
                - total_generation_time: Total time spent generating
        """
        from generation.chunking import split_into_sentences, estimate_duration
        from audio.streaming import StreamingAudioWriter

        # Estimate if text needs chunking
        estimated_duration = estimate_duration(text)
        print(f"\n[Long-form] Estimated duration: {estimated_duration:.1f}s for text length: {len(text)} chars")

        # Split into chunks
        chunks = split_into_sentences(text, max_duration_seconds=max_chunk_duration)
        print(f"[Long-form] Split into {len(chunks)} chunks")

        if len(chunks) == 1:
            print("[Long-form] Single chunk - using standard generation")

        # Generate each chunk with voice prefix for consistency
        audio_segments = []
        chunks_info = []
        total_generation_time = 0

        for i, chunk in enumerate(chunks):
            print(f"\n[Long-form] Generating chunk {i+1}/{len(chunks)}: '{chunk[:50]}...'")

            # Add voice prefix for consistency
            prompt = f"{voice}: {chunk}"

            # Create audio writer for this chunk
            audio_writer = StreamingAudioWriter(
                player,
                output_file=None,  # Don't write to file
                chunk_size=25,     # Use default chunk size
                lookback_frames=15  # Use default lookback
            )
            audio_writer.start()

            # Generate this chunk
            result = await self._generate_async(prompt, audio_writer, max_tokens=max_tokens)

            # Finalize and get audio
            audio = audio_writer.finalize()

            if audio is not None and len(audio) > 0:
                audio_segments.append(audio)
                chunks_info.append({
                    'chunk_index': i,
                    'text': chunk,
                    'duration': result['audio_duration'],
                    'generation_time': result['generation_time'],
                    'rtf': result['rtf']
                })
                total_generation_time += result['generation_time']
            else:
                print(f"[Long-form] Warning: No audio generated for chunk {i+1}")

        # Concatenate audio segments with silence
        if len(audio_segments) == 0:
            raise ValueError("No audio was generated")

        if len(audio_segments) == 1:
            final_audio = audio_segments[0]
        else:
            final_audio = self._concatenate_with_silence(
                audio_segments,
                silence_duration=silence_duration
            )

        total_duration = len(final_audio) / SAMPLE_RATE

        print(f"\n[Long-form] Complete!")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Total generation time: {total_generation_time:.2f}s")
        print(f"  Overall RTF: {total_generation_time / total_duration:.3f}")

        return {
            'audio': final_audio,
            'chunks_info': chunks_info,
            'total_duration': total_duration,
            'total_generation_time': total_generation_time,
            'num_chunks': len(chunks)
        }

    def _concatenate_with_silence(self, audio_segments, silence_duration=0.2):
        """Concatenate audio segments with short silence between them

        Args:
            audio_segments: List of numpy audio arrays
            silence_duration: Duration of silence in seconds

        Returns:
            Concatenated audio as numpy array
        """
        if len(audio_segments) == 1:
            return audio_segments[0]

        # Create silence buffer (zeros)
        silence_samples = int(silence_duration * SAMPLE_RATE)
        silence = np.zeros(silence_samples, dtype=audio_segments[0].dtype)

        # Concatenate segments with silence in between
        result = audio_segments[0]
        for next_segment in audio_segments[1:]:
            result = np.concatenate([result, silence, next_segment])

        return result
