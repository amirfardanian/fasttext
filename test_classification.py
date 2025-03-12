from capture_model.scoring import classify_lines

test_lines = ["afrunding", "bager søndag", "random unknown line"]
results = classify_lines(test_lines)

for res in results:
    print(res)
