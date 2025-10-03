#!/usr/bin/env python3
"""
Enhanced API Server Demo

This script demonstrates how to use the enhanced vLLM API server with token decoding features.
Run this after starting the enhanced API server (08_enhanced_api_server.py).
"""

import requests
import json
import sys
import time


def test_enhanced_api_server():
    """Test the enhanced API server with token decoding features."""
    
    base_url = "http://localhost:8000"
    
    print("🔤 Enhanced vLLM API Server Demo")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/")
        server_info = response.json()
        print(f"✅ Server running: {server_info['message']}")
        print(f"📦 Model: {server_info['model']}")
        print()
    except requests.exceptions.ConnectionError:
        print("❌ Server not running! Please start the enhanced API server first:")
        print("   python examples/08_enhanced_api_server.py")
        return False
    
    # Test 1: Basic completion with token decoding
    print("🔹 Test 1: Basic Completion with Token Decoding")
    print("-" * 50)
    
    completion_request = {
        "model": "gpt2",
        "prompt": "The benefits of renewable energy include",
        "max_tokens": 25,
        "temperature": 0.7,
        "include_tokens": True,
        "include_token_breakdown": True
    }
    
    response = requests.post(f"{base_url}/v1/completions", json=completion_request)
    result = response.json()
    
    print(f"Prompt: '{completion_request['prompt']}'")
    print(f"Generated: '{result['choices'][0]['text']}'")
    print(f"Token count: {result['choices'][0]['token_info']['completion_tokens']}")
    print("Token breakdown:")
    for token in result['choices'][0]['token_breakdown'][:10]:  # Show first 10
        print(f"  {token['index']:2d}: ID {token['token_id']:5d} = '{token['text']}'")
    print()
    
    # Test 2: Chat completion with token info
    print("🔹 Test 2: Chat Completion with Token Info")
    print("-" * 50)
    
    chat_request = {
        "model": "gpt2",
        "messages": [
            {"role": "user", "content": "Explain machine learning in simple terms"}
        ],
        "max_tokens": 30,
        "include_tokens": True
    }
    
    response = requests.post(f"{base_url}/v1/chat/completions", json=chat_request)
    result = response.json()
    
    print(f"User: {chat_request['messages'][0]['content']}")
    print(f"Assistant: {result['choices'][0]['message']['content']}")
    print(f"Tokens used: {result['token_info']['total_tokens']} (prompt: {result['token_info']['prompt_tokens']}, completion: {result['token_info']['completion_tokens']})")
    print(f"Token IDs: {result['token_info']['completion_token_ids'][:10]}...")  # Show first 10
    print()
    
    # Test 3: Token decoding endpoint
    print("🔹 Test 3: Direct Token Decoding")
    print("-" * 50)
    
    # Use some token IDs from previous response
    token_ids = result['token_info']['completion_token_ids'][:8]  # First 8 tokens
    
    decode_request = {
        "token_ids": token_ids,
        "skip_special_tokens": True
    }
    
    response = requests.post(f"{base_url}/v1/tokens/decode", json=decode_request)
    decode_result = response.json()
    
    print(f"Token IDs: {token_ids}")
    print(f"Decoded text: '{decode_result['decoded_text']}'")
    print("Individual tokens:")
    for token in decode_result['individual_tokens']:
        print(f"  {token['index']:2d}: ID {token['token_id']:5d} = '{token['text']}'")
    print()
    
    # Test 4: Special tokens info
    print("🔹 Test 4: Special Tokens Information")
    print("-" * 50)
    
    response = requests.get(f"{base_url}/v1/tokens/special")
    special_tokens = response.json()
    
    print(f"Model: {special_tokens['model']}")
    print(f"Vocabulary size: {special_tokens['vocab_size']:,}")
    print("Special tokens:")
    for token_type, info in special_tokens['special_tokens'].items():
        print(f"  {info['description']}: ID {info['id']} = '{info['text']}'")
    print()
    
    # Test 5: Streaming with token info
    print("🔹 Test 5: Streaming with Token Information")
    print("-" * 50)
    
    streaming_request = {
        "model": "gpt2",
        "prompt": "The future of technology will be",
        "max_tokens": 20,
        "stream": True,
        "include_tokens": True
    }
    
    print(f"Prompt: '{streaming_request['prompt']}'")
    print("Streaming response:")
    
    response = requests.post(
        f"{base_url}/v1/completions",
        json=streaming_request,
        stream=True
    )
    
    accumulated_text = ""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: ') and not line.startswith('data: [DONE]'):
                try:
                    chunk_data = json.loads(line[6:])  # Remove 'data: ' prefix
                    chunk_text = chunk_data['choices'][0]['text']
                    accumulated_text += chunk_text
                    
                    # Show token info if available
                    if 'token_info' in chunk_data:
                        token_info = chunk_data['token_info']
                        print(f"  Chunk: '{chunk_text}' (tokens: {token_info.get('new_tokens', [])}, total: {token_info.get('accumulated_tokens', 0)})")
                    else:
                        print(f"  Chunk: '{chunk_text}'")
                        
                except json.JSONDecodeError:
                    continue
    
    print(f"Final text: '{accumulated_text.strip()}'")
    print()
    
    print("✨ Enhanced API server demo completed!")
    print("\n💡 Key features demonstrated:")
    print("   - Token decoding in completion responses")
    print("   - Individual token breakdown")
    print("   - Special token information")
    print("   - Streaming with token details")
    print("   - Direct token encode/decode endpoints")
    
    return True


def main():
    """Main function."""
    if not test_enhanced_api_server():
        sys.exit(1)


if __name__ == "__main__":
    main()

