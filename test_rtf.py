"""Quick test to verify RTF output with BnB quantization"""

import asyncio
from generation.vllm_generator import VLLMTTSGenerator
from audio import LLMAudioPlayer, StreamingAudioWriter
from config import CHUNK_SIZE, LOOKBACK_FRAMES, BNB_QUANTIZATION

async def main():
    print("Initializing VLLM generator with BnB quantization...")
    generator = VLLMTTSGenerator(
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        quantization=BNB_QUANTIZATION
    )

    # Initialize engine
    await generator.initialize_engine()

    player = LLMAudioPlayer(generator.tokenizer)

    # Short test prompt
    # prompt = "Hello world, testing real time factor."
    prompt = "Shanghai is a direct-administered municipality and the most populous urban area in China."

    audio_writer = StreamingAudioWriter(
        player,
        output_file=None,
        chunk_size=CHUNK_SIZE,
        lookback_frames=LOOKBACK_FRAMES
    )
    audio_writer.start()

    # Generate
    result = await generator._generate_async(prompt, audio_writer)
    audio_writer.finalize()

    # Print results
    print(f"\nResults:")
    print(f"  Tokens: {len(result['all_token_ids'])}")
    print(f"  Audio duration: {result['audio_duration']:.2f}s")
    print(f"  Generation time: {result['generation_time']:.2f}s")
    print(f"  RTF: {result['rtf']:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
