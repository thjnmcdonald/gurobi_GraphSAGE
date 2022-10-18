location_list = ['2022-09-16_21:18:03-GraphSAGE_16_3',
 '2022-09-16_21:20:47-GraphSAGE_16_3',
 '2022-09-16_21:18:34-GraphSAGE_16_3',
 '2022-09-16_21:19:45-GraphSAGE_16_3',
 '2022-09-16_21:20:37-GraphSAGE_16_3']

def create_bash_script(location_list):
    print('SECONDS=0')
    print('echo \"started!\"')
    print(f'now=$(date +"%T")')
    print(f'echo \"Started at : $now\"')

    mol_len_list = [4]
    for j, mol_len in enumerate(mol_len_list):
        for i, location in enumerate(location_list):
            # note that I added extra_runs here to the python file name
            print(f'python GraphSAGE_MILP_extra_runs.py --location {location} --time_lim 36000 --mol_len {mol_len} > GraphSAGE_outputs/{location}_mol_len_{mol_len}.txt')
            print(f'echo \"progress: {(j)*len(location_list) + i + 1}/{(len(location_list))*(len(mol_len_list))} after: $SECONDS s\"')
            print(f'SECONDS=0')
            print(f'now=$(date +"%T")')
            print(f'echo \"Started at : $now\"')

create_bash_script(location_list)

# run as
# python bash_script_creator.py > bash.script