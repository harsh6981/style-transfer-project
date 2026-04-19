import os
from style_transfer import (
    load_image,
    save_image,
    run_style_transfer,
    cnn,
    cnn_normalization_mean,
    cnn_normalization_std
)

# List of test cases
test_cases = [
    ("content_images/img1.jpg.jpg", "style_images/style_images1.jpg", "output_images/output1.jpg"),
    ("content_images/img2.jpg.jpg", "style_images/style_images2.jpg", "output_images/output2.jpg"),
    ("content_images/img3.jpg.jpg", "style_images/style_images3.jpg", "output_images/output3.jpg"),
    ("content_images/img4.jpg.jpg", "style_images/style_images4.jpg", "output_images/output4.jpg"),
    ("content_images/img5.jpg.jpg", "style_images/style_images5.jpg", "output_images/output5.jpg"),
]

os.makedirs("output_images", exist_ok=True)

for idx, (content_path, style_path, output_path) in enumerate(test_cases, start=1):
    print(f"\nRunning Test Case {idx}")
    print(f"Content: {content_path}")
    print(f"Style  : {style_path}")

    content_img = load_image(content_path)
    style_img = load_image(style_path)
    input_img = content_img.clone()

    output = run_style_transfer(
        cnn,
        cnn_normalization_mean,
        cnn_normalization_std,
        content_img,
        style_img,
        input_img,
        num_steps=300
    )

    save_image(output, output_path)
    print(f"Saved output to: {output_path}")

print("\nAll test cases completed successfully.")