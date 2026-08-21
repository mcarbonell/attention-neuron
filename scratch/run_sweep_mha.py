from test_mha_onthefly import test_mha_onthefly

print("Testing n_pairs=8, seq_len=128:")
test_mha_onthefly(128, 8, lr=3e-3, target_steps=1000)

print("\nTesting n_pairs=16, seq_len=128:")
test_mha_onthefly(128, 16, lr=3e-3, target_steps=1000)

print("\nTesting n_pairs=32, seq_len=128:")
test_mha_onthefly(128, 32, lr=3e-3, target_steps=1000)
