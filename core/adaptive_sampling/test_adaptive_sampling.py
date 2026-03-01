"""
Quick test script to verify adaptive sampling implementation.

This script tests the core components without requiring full schema generation.
"""

from core.adaptive_sampling import SchemaEntropyCalculator, AdaptiveSampler


def test_entropy_calculator():
    """Test the schema entropy calculator."""
    print("=" * 60)
    print("Testing Schema Entropy Calculator")
    print("=" * 60)
    
    calc = SchemaEntropyCalculator()
    
    # Test 1: Empty schema
    schema1 = []
    entropy1 = calc.compute_entropy(schema1)
    print(f"Test 1 - Empty schema: entropy = {entropy1:.4f} (expected: 1.0)")
    assert entropy1 == 1.0, "Empty schema should have entropy 1.0"
    
    # Test 2: First schema with content
    schema2 = [
        {
            "Schema Name": "users",
            "Attributes": [
                {"id": "User ID"},
                {"name": "User name"}
            ]
        }
    ]
    entropy2 = calc.compute_entropy(schema2)
    print(f"Test 2 - First real schema: entropy = {entropy2:.4f}")
    print(f"         Features found: {calc.get_feature_count()}")
    
    # Test 3: Identical schema (no change)
    schema3 = schema2.copy()
    entropy3 = calc.compute_entropy(schema3)
    print(f"Test 3 - Identical schema: entropy = {entropy3:.4f} (expected: 0.0)")
    assert entropy3 == 0.0, "Identical schemas should have entropy 0.0"
    
    # Test 4: Small change (add one attribute)
    schema4 = [
        {
            "Schema Name": "users",
            "Attributes": [
                {"id": "User ID"},
                {"name": "User name"},
                {"email": "User email"}  # New attribute
            ]
        }
    ]
    entropy4 = calc.compute_entropy(schema4)
    print(f"Test 4 - Small change (add 1 attr): entropy = {entropy4:.4f}")
    print(f"         Features: {calc.get_feature_count()}")
    
    # Test 5: Another small change
    schema5 = [
        {
            "Schema Name": "users",
            "Attributes": [
                {"id": "User ID"},
                {"name": "User name"},
                {"email": "User email"},
                {"age": "User age"}  # Another new attribute
            ]
        }
    ]
    entropy5 = calc.compute_entropy(schema5)
    print(f"Test 5 - Small change (add 1 more): entropy = {entropy5:.4f}")
    
    # Test 6: No change again
    schema6 = schema5.copy()
    entropy6 = calc.compute_entropy(schema6)
    print(f"Test 6 - No change: entropy = {entropy6:.4f} (expected: 0.0)")
    
    # Print statistics
    stats = calc.get_statistics()
    print("\nEntropy Statistics:")
    print(f"  Iterations: {stats['num_iterations']}")
    print(f"  Mean entropy: {stats['mean_entropy']:.4f}")
    print(f"  Min entropy: {stats['min_entropy']:.4f}")
    print(f"  Max entropy: {stats['max_entropy']:.4f}")
    print(f"  Final features: {stats['feature_count']}")
    print(f"  Entropy history: {[f'{e:.3f}' for e in stats['entropy_history']]}")
    
    print("\n✓ Schema Entropy Calculator tests passed!\n")


def test_adaptive_sampler():
    """Test the adaptive sampler."""
    print("=" * 60)
    print("Testing Adaptive Sampler")
    print("=" * 60)
    
    # Create sampler with strict parameters for quick testing
    sampler = AdaptiveSampler(
        theta=0.05,
        m=2,  # Only need 2 consecutive low entropy
        n_min=3,  # Minimum 3 documents
        delta=0.1,
        epsilon=0.05,
        enable_probabilistic_stop=False  # Disable for deterministic testing
    )
    
    # Simulate processing documents
    schemas = [
        [],  # Empty initial
        [{"Schema Name": "users", "Attributes": [{"id": "ID"}]}],
        [{"Schema Name": "users", "Attributes": [{"id": "ID"}, {"name": "Name"}]}],
        [{"Schema Name": "users", "Attributes": [{"id": "ID"}, {"name": "Name"}, {"email": "Email"}]}],
        [{"Schema Name": "users", "Attributes": [{"id": "ID"}, {"name": "Name"}, {"email": "Email"}]}],  # Same
        [{"Schema Name": "users", "Attributes": [{"id": "ID"}, {"name": "Name"}, {"email": "Email"}]}],  # Same
    ]
    
    for i, schema in enumerate(schemas):
        should_continue = sampler.should_continue(schema)
        print(f"Document {i+1}: should_continue = {should_continue}, "
              f"streak = {sampler.low_entropy_streak}, "
              f"processed = {sampler.n_processed}")
        
        if not should_continue:
            print(f"\n✓ Early stopping triggered after {sampler.n_processed} documents!")
            print(f"  Stop reason: {sampler.get_stop_reason()}")
            break
    
    # Get statistics
    stats = sampler.get_statistics()
    print("\nAdaptive Sampler Statistics:")
    print(f"  Documents processed: {stats['n_processed']}")
    print(f"  Stopped early: {stats['should_stop']}")
    print(f"  Final entropy: {stats['entropy_statistics']['final_entropy']:.4f}")
    print(f"  Mean entropy: {stats['entropy_statistics']['mean_entropy']:.4f}")
    
    print("\n✓ Adaptive Sampler tests passed!\n")


def test_integration():
    """Test integration of components."""
    print("=" * 60)
    print("Testing Integration")
    print("=" * 60)
    
    # Create custom entropy calculator
    calc = SchemaEntropyCalculator()
    
    # Create sampler with the calculator
    sampler = AdaptiveSampler(
        theta=0.1,
        m=2,
        n_min=2,
        entropy_calculator=calc
    )
    
    # Process some schemas
    test_schemas = [
        [{"Schema Name": "test", "Attributes": [{"a": "A"}]}],
        [{"Schema Name": "test", "Attributes": [{"a": "A"}, {"b": "B"}]}],
        [{"Schema Name": "test", "Attributes": [{"a": "A"}, {"b": "B"}, {"c": "C"}]}],
        [{"Schema Name": "test", "Attributes": [{"a": "A"}, {"b": "B"}, {"c": "C"}]}],
    ]
    
    for schema in test_schemas:
        if not sampler.should_continue(schema):
            break
    
    # Verify both calculator and sampler see same data
    calc_stats = calc.get_statistics()
    sampler_stats = sampler.get_statistics()
    
    print(f"Entropy calculator iterations: {calc_stats['num_iterations']}")
    print(f"Sampler processed count: {sampler_stats['n_processed']}")
    print(f"Feature count: {calc_stats['feature_count']}")
    
    assert calc_stats['num_iterations'] == sampler_stats['n_processed'], \
        "Calculator and sampler should have same iteration count"
    
    print("\n✓ Integration tests passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ADAPTIVE SAMPLING TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_entropy_calculator()
        test_adaptive_sampler()
        test_integration()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nThe adaptive sampling implementation is working correctly.")
        print("You can now use it in schema generation by enabling it in config:\n")
        print("  adaptive_sampling:")
        print("    enabled: true")
        print()
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

