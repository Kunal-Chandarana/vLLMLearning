#!/usr/bin/env python3
"""
Token Decoding Example

This example demonstrates how to decode completion tokens back to English text.
Features:
- Basic token-to-text conversion
- Individual token analysis
- Streaming token decoding simulation
- API response integration
- Special token handling
- Round-trip encoding/decoding
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_model_parallel


class TokenDecodingExample:
    """Demonstrates various token decoding techniques in vLLM."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.llm = None
        self.tokenizer = None
        
    def initialize_model(self):
        """Initialize the vLLM model and tokenizer."""
        print(f"📦 Loading model: {self.model_name}")
        
        try:
            # Initialize vLLM engine
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=1,
                max_model_len=512,
                gpu_memory_utilization=0.8,
                trust_remote_code=True,
            )
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Model and tokenizer loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Try using a smaller model like 'gpt2' or 'distilgpt2'")
            return False
    
    def basic_token_decoding(self):
        """Demonstrate basic token-to-text decoding."""
        print("\n🔹 Basic Token Decoding")
        print("-" * 50)
        
        # Generate some text
        prompt = "The future of artificial intelligence is"
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=30,
            stop=["\n"]  # Stop at newlines for cleaner output
        )
        
        print(f"Prompt: '{prompt}'")
        
        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        # Extract generated data
        generated_text = output.outputs[0].text
        token_ids = output.outputs[0].token_ids
        prompt_token_ids = output.prompt_token_ids
        
        print(f"Generated text: '{generated_text}'")
        print(f"Generated token IDs: {token_ids}")
        print(f"Token count: {len(token_ids)}")
        print()
        
        # Decode tokens back to text
        decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"🔄 Decoded from tokens: '{decoded_text}'")
        print(f"✅ Perfect match? {generated_text.strip() == decoded_text.strip()}")
        print()
        
        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'token_ids': token_ids,
            'decoded_text': decoded_text,
            'prompt_tokens': len(prompt_token_ids),
            'completion_tokens': len(token_ids)
        }
    
    def individual_token_analysis(self, token_ids: List[int], max_tokens: int = 15):
        """Analyze individual tokens in detail."""
        print("🔹 Individual Token Analysis")
        print("-" * 50)
        
        print(f"{'Token':<6} {'ID':<8} {'Raw Text':<15} {'Clean Text':<15}")
        print("-" * 60)
        
        for i, token_id in enumerate(token_ids[:max_tokens]):
            # Decode single token with and without special tokens
            raw_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            clean_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            
            # Handle display of special characters
            raw_display = repr(raw_text)[1:-1] if raw_text != clean_text else raw_text
            clean_display = repr(clean_text)[1:-1] if '\n' in clean_text else clean_text
            
            print(f"{i:<6} {token_id:<8} {raw_display:<15} {clean_display:<15}")
        
        if len(token_ids) > max_tokens:
            print(f"... and {len(token_ids) - max_tokens} more tokens")
        print()
    
    def round_trip_encoding_test(self):
        """Test encoding text to tokens and decoding back."""
        print("🔹 Round-trip Encoding/Decoding Test")
        print("-" * 50)
        
        test_texts = [
            "Hello, world! How are you today?",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming technology.",
            "Special characters: @#$%^&*()_+-={}[]|\\:;\"'<>?,./"
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"Test {i}: '{text}'")
            
            # Encode to tokens
            tokens = self.tokenizer.encode(text)
            print(f"  Tokens: {tokens}")
            
            # Decode back to text
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"  Decoded: '{decoded}'")
            
            # Check if round trip is perfect
            is_perfect = text == decoded
            print(f"  ✅ Perfect round trip: {is_perfect}")
            
            if not is_perfect:
                print(f"  ⚠️  Difference detected!")
            print()
    
    def simulate_streaming_decoding(self):
        """Simulate how token decoding works in streaming scenarios."""
        print("🔹 Streaming Token Decoding Simulation")
        print("-" * 50)
        
        # Generate some tokens first
        prompt = "The benefits of renewable energy include"
        sampling_params = SamplingParams(temperature=0.7, max_tokens=20)
        
        outputs = self.llm.generate([prompt], sampling_params)
        streaming_tokens = outputs[0].outputs[0].token_ids
        
        print(f"Simulating streaming for: '{prompt}'")
        print("Streaming token-by-token decoding:")
        print()
        
        accumulated_tokens = []
        previous_text = ""
        
        for i, token_id in enumerate(streaming_tokens):
            accumulated_tokens.append(token_id)
            
            # Decode current accumulated tokens
            current_full_text = self.tokenizer.decode(accumulated_tokens, skip_special_tokens=True)
            
            # Calculate the delta (new text)
            new_text = current_full_text[len(previous_text):]
            
            # Single token text
            single_token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            
            print(f"Step {i+1:2d}: Token {token_id:5d} | Single: '{single_token_text}' | Delta: '{new_text}' | Full: '{current_full_text}'")
            
            previous_text = current_full_text
        
        print(f"\n✨ Streaming complete! Final text: '{current_full_text}'")
        print()
    
    def special_token_handling(self):
        """Demonstrate handling of special tokens in decoding."""
        print("🔹 Special Token Handling")
        print("-" * 50)
        
        # Create text with special tokens
        if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
            text_with_special = f"Hello{self.tokenizer.eos_token}World{self.tokenizer.eos_token}End"
        else:
            # Fallback for tokenizers without explicit eos_token
            text_with_special = "Hello<|endoftext|>World<|endoftext|>End"
        
        tokens = self.tokenizer.encode(text_with_special)
        
        print(f"Text with special tokens: '{text_with_special}'")
        print(f"Encoded tokens: {tokens}")
        print()
        
        # Decode with different options
        print("Decoding options:")
        keep_special = self.tokenizer.decode(tokens, skip_special_tokens=False)
        skip_special = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        print(f"  Keep special tokens:  '{keep_special}'")
        print(f"  Skip special tokens:  '{skip_special}'")
        print()
        
        # Show what each special token ID represents
        special_token_info = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': getattr(self.tokenizer, 'pad_token_id', None),
            'unk_token_id': getattr(self.tokenizer, 'unk_token_id', None),
            'bos_token_id': getattr(self.tokenizer, 'bos_token_id', None),
        }
        
        print("Special token meanings:")
        for token_name, token_id in special_token_info.items():
            if token_id is not None:
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                meaning = token_name.replace('_', ' ').title()
                print(f"  {meaning}: ID {token_id} = '{token_text}'")
        print()
    
    def api_response_integration(self):
        """Show how to integrate token decoding into API responses."""
        print("🔹 API Response Integration")
        print("-" * 50)
        
        # Generate response
        prompt = "Explain the importance of renewable energy"
        sampling_params = SamplingParams(temperature=0.7, max_tokens=40)
        
        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        generated_text = output.outputs[0].text
        token_ids = output.outputs[0].token_ids
        prompt_token_ids = output.prompt_token_ids
        
        # Create enhanced API response with token information
        api_response = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "text": generated_text,
                    "index": 0,
                    "finish_reason": "stop",
                    # Enhanced with token information
                    "token_ids": token_ids,
                    "decoded_verification": self.tokenizer.decode(token_ids, skip_special_tokens=True),
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(token_ids),
                "total_tokens": len(prompt_token_ids) + len(token_ids),
            },
            # Debug information
            "debug": {
                "individual_tokens": [
                    {
                        "id": token_id,
                        "text": self.tokenizer.decode([token_id], skip_special_tokens=True),
                        "raw": self.tokenizer.decode([token_id], skip_special_tokens=False)
                    }
                    for token_id in token_ids[:10]  # First 10 tokens
                ]
            }
        }
        
        print("Enhanced API Response with Token Information:")
        print(json.dumps(api_response, indent=2))
        print()
    
    def batch_token_decoding(self):
        """Demonstrate batch token decoding."""
        print("🔹 Batch Token Decoding")
        print("-" * 50)
        
        prompts = [
            "The weather today is",
            "My favorite programming language is",
            "The key to success is"
        ]
        
        sampling_params = SamplingParams(temperature=0.7, max_tokens=15)
        batch_outputs = self.llm.generate(prompts, sampling_params)
        
        print("Batch generation and token decoding:")
        print()
        
        for i, output in enumerate(batch_outputs):
            prompt = output.prompt
            generated = output.outputs[0].text
            tokens = output.outputs[0].token_ids
            
            # Decode tokens
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
            
            print(f"Batch {i+1}:")
            print(f"  Prompt: '{prompt}'")
            print(f"  Generated: '{generated}'")
            print(f"  From tokens: '{decoded}'")
            print(f"  Token IDs: {tokens}")
            print(f"  Token count: {len(tokens)}")
            print()
    
    def cleanup(self):
        """Clean up resources."""
        try:
            destroy_model_parallel()
        except:
            pass


def main():
    """Main function to run token decoding examples."""
    print("🔤 vLLM Token Decoding Examples")
    print("=" * 60)
    
    # Initialize example class
    example = TokenDecodingExample(model_name="gpt2")  # Using GPT-2 for reliability
    
    try:
        # Initialize model
        if not example.initialize_model():
            return
        
        # Run all examples
        print("\n🚀 Running Token Decoding Examples...")
        
        # 1. Basic decoding
        result = example.basic_token_decoding()
        
        # 2. Individual token analysis
        example.individual_token_analysis(result['token_ids'])
        
        # 3. Round-trip test
        example.round_trip_encoding_test()
        
        # 4. Streaming simulation
        example.simulate_streaming_decoding()
        
        # 5. Special token handling
        example.special_token_handling()
        
        # 6. API integration
        example.api_response_integration()
        
        # 7. Batch decoding
        example.batch_token_decoding()
        
        print("✨ All token decoding examples completed!")
        print("\n💡 Key takeaways:")
        print("   - Use tokenizer.decode(token_ids) to convert tokens back to text")
        print("   - skip_special_tokens=True removes <|endoftext|> and similar tokens")
        print("   - Individual tokens can be decoded separately for analysis")
        print("   - Perfect round-trip encoding/decoding is usually possible")
        print("   - Streaming requires accumulating tokens progressively")
        print("   - Token information can enhance API responses for debugging")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        example.cleanup()


if __name__ == "__main__":
    main()

