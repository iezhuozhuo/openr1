from datasets import load_dataset
import sys
sys.path.append("/extrahome0/Zhuo/code/open-r1")
from src.evol.fitness import extract_answer

# data = load_dataset('/extrahome0/HF_datasets//NuminaMath-TIR')
# for split, split_dataset in data.items():
#     split_dataset.to_json(f'/code/Research_with_zhuo/reasoning/GA/NuminaMath-TIR-{split}.json')


ds = load_dataset("simplescaling/s1K")["train"]
cnt = 0
gt_cnt = 0
for idx in range(len(ds)):
    data = ds[idx]["metadata"]
    gt_ans = extract_answer(ds[idx]["attempt"])
    if gt_ans:
        gt_cnt += 1
    else:
        data = eval(ds[idx]["metadata"])
        if "answer" in data:
            cnt += 1
            print(data["answer"])
print(gt_cnt, cnt, len(ds))
#

# from src.open_r1.rewards import accuracy_reward, format_reward
# sol = "To find the side length of the square inscribed in the right triangle with legs of lengths 6 and 8, we can follow these steps:\n\n1. **Understand the problem:**\n   - We have a right triangle with legs of lengths \\(a = 6\\) and \\(b = 8\\).\n   - A square is inscribed such that it shares the right angle of the triangle.\n\n2. **Visualize the problem:**\n   - The square will have one of its sides along each leg of the triangle and its hypotenuse parallel to the hypotenuse of the triangle.\n\n3. **Formulate the problem:**\n   - Let the side length of the square be \\(s\\).\n   - The square will divide each leg into two segments: one segment equal to \\(s\\) and the remaining part of the leg.\n   - For the leg \\(a\\), the remaining part will be \\(6 - s\\).\n   - For the leg \\(b\\), the remaining part will be \\(8 - s\\).\n   \n4. **Use geometry:**\n   - The triangle formed by the remaining parts of the original triangle and the side of the square is similar to the original triangle.\n   - The legs of the smaller triangle are \\(6 - s\\) and \\(8 - s\\).\n\n5. **Set up the equation:**\n   - From the similarity of the triangles, \\(\\frac{6 - s}{s} = \\frac{s}{8 - s}\\).\n\n6. **Solve the equation for \\(s\\):**\n\nLet's implement this in Python using SymPy to find the side length \\(s\\) of the square.\n\n```python\nimport sympy as sp\n\n# Define the variable\ns = sp.symbols('s')\n\n# Define the equation based on the similarity of triangles\nequation = sp.Eq((6 - s) / s, s / (8 - s))\n\n# Solve the equation for s\nsolution = sp.solve(equation, s)\n\n# Filter the valid solution (positive value)\nvalid_solution = [sol for sol in solution if sol > 0]\n\n# Print the valid side length of the square\nprint(valid_solution[0])\n```\n```output\n24/7\n```\nThe side length of the square inscribed in the right triangle with legs of lengths 6 and 8 is \\(\\boxed{\\frac{24}{7}}\\)."
#
# print(accuracy_reward([[{"content": sol}]], ['\\frac{24}{7}']))
