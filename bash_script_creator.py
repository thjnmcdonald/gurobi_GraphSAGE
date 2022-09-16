location_list = ['2022-09-16_21:00:03-GraphSAGE_32_1',
 '2022-09-16_21:09:07-GraphSAGE_16_2',
 '2022-09-16_21:12:46-GraphSAGE_32_2',
 '2022-09-16_21:16:04-GraphSAGE_64_2',
 '2022-09-16_21:18:55-GraphSAGE_16_3',
 '2022-09-16_21:26:00-GraphSAGE_32_3',
 '2022-09-16_21:29:19-GraphSAGE_64_3']

def create_bash_script(location_list):
    print('SECONDS=0')
    print('echo \"started!\"')
    print(f'now=$(date +"%T")')
    print(f'echo \"Started at : $now\"')

    mol_len_list = [4, 6, 8]
    for j, mol_len in enumerate(mol_len_list):
        for i, location in enumerate(location_list):
            print(f'python GraphSAGE_MILP.py --location {location} --time_lim 10800 --mol_len {mol_len} > GraphSAGE_outputs/{location[-14:]}_mol_len_{mol_len}.txt')
            print(f'echo \"progress: {(j)*len(location_list) + i + 1}/{(len(location_list))*(len(mol_len_list))} after: $SECONDS s\"')
            print(f'SECONDS=0')
            print(f'now=$(date +"%T")')
            print(f'echo \"Started at : $now\"')

create_bash_script(location_list)

# run as
# python bash_script_creator.py > bash.script