def extract_aig_data(file_path):
    # Open the text file for reading
    with open(file_path, 'r') as file:
        data = file.readlines()  # Read lines into a list
    
    # Find the line starting with "Name:"
    results = []
    # Extract the line
    for line in data:
        # print(line)
        if line.startswith("Name:"):
            parts = line.split()
    # Parse the extracted line
            # parts = line.split(' ')
            name = parts[1]
            # initial_aig = parts[4]
            try:
                initial_aig = parts[4]
                final_gt_aig = parts[12]
            except:
                name = 'b9_.aig'
                initial_aig = parts[3]
                final_gt_aig = -float('inf')
            # print(f'Name: {name}, Initial AIG: {initial_aig}, Final gt AIG: {final_gt_aig}')
            # raise ValueError
            # Return the extracted values
            results.append({
                'name': name,
                'initial_aig': initial_aig,
                'final_gt_aig': final_gt_aig
            }
            )
            # print(results[-1])
    return results 

# Example usage
# file_path = './gnn_now_gnn_future/over_method_BFS_step_4_maxsize_200_2024-06-06_22-59-34' + '.txt'
file_path = './gnn_now_gnn_future_finetuned/' + 'method_BestFirstSearch_step_10_maxsize_10_2024-06-08_14-48-10' + '.txt'
result = extract_aig_data(file_path)
# print(result)

sequence = [
    'alu4',
    'apex1',
    'apex2',
    'apex4',
    'b9',
    'bar',
    'c7552',
    'c880',
    'cavlc',
    'div',
    'i9',
    'm4',
    'max1024',
    'memctrl',
    'pair',
    'prom',
    'router',
    'sqrt',
    'square',
    'voter',
]

for element in sequence:
    flag = False 
    for dat in result:
        if element in dat['name']:
            print(f"{dat['name']} & {max(float(dat['initial_aig']),float(dat['final_gt_aig']))}")
            flag = True 
            break
    if not flag:
        print(f"{element} & - & -")