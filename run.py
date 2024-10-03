import subprocess, os

output_dir = "./output/eduAgent"
os.makedirs(output_dir, exist_ok=True)

def run_all_experiments():
    model_names = ["akt", "atkt", "dkt", "dkvmn", "simpleKT"]
    input_types = ["past", "past_future"]

    for model_name in model_names:
        for input_type in input_types:
            output_file = os.path.join(output_dir, f"{model_name}_{input_type}.txt")
            print(f"Running experiment with model_name={model_name} and input_type={input_type}. Output will be saved to {output_file}")
            with open(output_file, "w") as f:
                subprocess.run(
                    ["python", "pipeline_v2.py", "--model_name", model_name, "--input_type", input_type],
                    stdout=f,
                    stderr=f
                )

if __name__ == "__main__":
    run_all_experiments()