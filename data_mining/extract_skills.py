# Read .csv file
csv_file = 'LI_skills.csv'
skills = []
try:
    with open(csv_file, 'r') as file:
        for line in file:
            skills.append(line.strip().lower())
except FileNotFoundError as e:
    print(f"File not found: {csv_file}")

# Pooling the skills
skill_count = {}
for skill in skills:
    if skill in skill_count:
        skill_count[skill] += 1
    else:
        skill_count[skill] = 1

# Sorting the skills based on their counts (descending)
skill_count = sorted(skill_count.items(), key=lambda item: item[1], reverse=True)
skill_count = skill_count[1:]

# Save the results in to a .txt file
n_top = 50
file_path = f'top{n_top}_LIskills.txt'
try:
    with open(file_path, 'w') as file:
        for item in skill_count[:n_top]:
            file.write(str(item) + '\n')
    print(f'The output has been saved to {file_path}')
except PermissionError as e:
    print(f"Permission denied to write to {file_path}")
except Exception as e:
    print(f"An error occurred while writing to the file: {e}")
