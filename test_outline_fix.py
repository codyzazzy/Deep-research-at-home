#!/usr/bin/env python3
"""
Test script to verify the outline processing fix.
This script tests the _flatten_outline_topics method to ensure it properly handles
nested outline structures and returns only strings.
"""

import sys
import os

# Add the current directory to the path so we can import the pipe
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the pipe class
try:
    from importlib import import_module
    pipe_module = import_module("Deep research at home")
    Pipe = pipe_module.Pipe
except ImportError as e:
    print(f"Error importing pipe: {e}")
    sys.exit(1)

def test_flatten_outline_topics():
    """Test the _flatten_outline_topics method with various outline structures"""
    
    # Create a pipe instance
    pipe = Pipe()
    
    # Test case 1: Simple outline with string subtopics
    print("=== Test Case 1: Simple outline with string subtopics ===")
    outline1 = [
        {
            "topic": "Introduction to AI",
            "subtopics": ["History of AI", "Basic Concepts", "Current Applications"]
        },
        {
            "topic": "Machine Learning",
            "subtopics": ["Supervised Learning", "Unsupervised Learning"]
        }
    ]
    
    result1 = pipe._flatten_outline_topics(outline1)
    print(f"Input: {outline1}")
    print(f"Output: {result1}")
    print(f"All strings: {all(isinstance(item, str) for item in result1)}")
    print(f"Length: {len(result1)}")
    print()
    
    # Test case 2: Nested outline with dict subtopics (the problematic case)
    print("=== Test Case 2: Nested outline with dict subtopics ===")
    outline2 = [
        {
            "topic": "Deep Learning",
            "subtopics": [
                {
                    "topic": "Neural Networks",
                    "subtopics": ["Perceptrons", "Backpropagation"]
                },
                {
                    "topic": "CNN",
                    "subtopics": ["Convolution", "Pooling"]
                }
            ]
        }
    ]
    
    result2 = pipe._flatten_outline_topics(outline2)
    print(f"Input: {outline2}")
    print(f"Output: {result2}")
    print(f"All strings: {all(isinstance(item, str) for item in result2)}")
    print(f"Length: {len(result2)}")
    print()
    
    # Test case 3: Mixed outline (strings and dicts)
    print("=== Test Case 3: Mixed outline ===")
    outline3 = [
        {
            "topic": "AI Ethics",
            "subtopics": [
                "Bias in AI",
                {
                    "topic": "Privacy Concerns",
                    "subtopics": ["Data Collection", "User Consent"]
                },
                "Transparency"
            ]
        }
    ]
    
    result3 = pipe._flatten_outline_topics(outline3)
    print(f"Input: {outline3}")
    print(f"Output: {result3}")
    print(f"All strings: {all(isinstance(item, str) for item in result3)}")
    print(f"Length: {len(result3)}")
    print()
    
    # Test case 4: Empty and edge cases
    print("=== Test Case 4: Edge cases ===")
    outline4 = []
    result4 = pipe._flatten_outline_topics(outline4)
    print(f"Empty outline: {result4}")
    
    outline5 = [{"topic": "Single Topic", "subtopics": []}]
    result5 = pipe._flatten_outline_topics(outline5)
    print(f"Single topic, no subtopics: {result5}")
    print(f"All strings: {all(isinstance(item, str) for item in result5)}")
    print()
    
    return all([
        all(isinstance(item, str) for item in result1),
        all(isinstance(item, str) for item in result2),
        all(isinstance(item, str) for item in result3),
        all(isinstance(item, str) for item in result4),
        all(isinstance(item, str) for item in result5),
    ])

def test_old_vs_new_behavior():
    """Simulate the old problematic behavior vs new fixed behavior"""
    
    print("=== Simulating Old vs New Behavior ===")
    
    # This is what the old code would have done (problematic)
    outline_items = [
        {
            "topic": "Main Topic",
            "subtopics": [
                {
                    "topic": "Subtopic 1",
                    "subtopics": ["Sub-sub 1", "Sub-sub 2"]
                },
                "String Subtopic"
            ]
        }
    ]
    
    print("Old behavior simulation:")
    flat_items_old = []
    for topic_item in outline_items:
        topic = topic_item.get("topic", "")
        subtopics = topic_item.get("subtopics", [])
        
        flat_items_old.append(topic)
        
        # This is the problematic part - adding dict objects directly
        for subtopic in subtopics:
            flat_items_old.append(subtopic)  # This would add dict objects!
    
    print(f"Old flat_items: {flat_items_old}")
    print(f"Types in old flat_items: {[type(item) for item in flat_items_old]}")
    print(f"Contains non-strings: {not all(isinstance(item, str) for item in flat_items_old)}")
    print()
    
    # New behavior using the fixed method
    pipe = Pipe()
    flat_items_new = pipe._flatten_outline_topics(outline_items)
    
    print("New behavior:")
    print(f"New flat_items: {flat_items_new}")
    print(f"Types in new flat_items: {[type(item) for item in flat_items_new]}")
    print(f"All strings: {all(isinstance(item, str) for item in flat_items_new)}")
    print()
    
    return all(isinstance(item, str) for item in flat_items_new)

if __name__ == "__main__":
    print("Testing outline processing fix...")
    print("=" * 60)
    
    try:
        # Test the flatten method
        test1_passed = test_flatten_outline_topics()
        print(f"Flatten outline test passed: {test1_passed}")
        print()
        
        # Test old vs new behavior
        test2_passed = test_old_vs_new_behavior()
        print(f"Old vs new behavior test passed: {test2_passed}")
        print()
        
        if test1_passed and test2_passed:
            print("✅ All tests passed! The fix should resolve the TypeError issue.")
        else:
            print("❌ Some tests failed. The fix may need additional work.")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
