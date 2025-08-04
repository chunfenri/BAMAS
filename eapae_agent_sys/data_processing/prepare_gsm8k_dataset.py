import json
import os
def get_difficulty(answer: str) -> str:
    """
    Determines the difficulty of a problem based on the number of calculation steps in the answer.
    """
    num_steps = answer.count('<<')
    if num_steps <= 1:
        return "direct"
    elif 2 <= num_steps <= 4:
        return "simple"
    else:
        return "complex"
def main():
    """
    Reads the GSM8K training data, adds a difficulty label based on the number of
    calculation steps, and saves the processed data.
    """
    input_file = "data/gsm8k/train.jsonl"
    output_dir = "data/processed"
    output_file = os.path.join(output_dir, "gsm8k_train_w_curriculum.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    processed_data = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            data = json.loads(line)
            difficulty = get_difficulty(data['answer'])
            data['difficulty'] = difficulty
            processed_data.append(data)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for data in processed_data:
            f_out.write(json.dumps(data) + '\n')
    print(f"Processed dataset saved to {output_file}")
    difficulty_counts = {}
    for data in processed_data:
        difficulty = data['difficulty']
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    for difficulty, count in difficulty_counts.items():
        print(f"- {difficulty}: {count} problems")
if __name__ == "__main__":
    main() 