# 生成TACRED小数据集
import json
tacred_dir = '../dataset/tacred/'
datasets = ['test', 'dev', 'train']

standford_berkeley = '_std_ber.json'
standford_berkeley_small = '_std_ber_small3.json'

for ds in datasets:
    print('small dataset ' + ds)
    datafile = tacred_dir + ds
    with open(datafile + standford_berkeley, 'r') as f:
        datas = list(json.load(f))

    new_datas = []
    for i in range(0, len(datas)//3):
        new_data = {}
        new_data['relation'] = datas[i]['relation']
        new_data['token'] = datas[i]['token']
        new_data['subj_start'] = datas[i]['subj_start']
        new_data['subj_end'] = datas[i]['subj_end']
        new_data['obj_start'] = datas[i]['obj_start']
        new_data['obj_end'] = datas[i]['obj_end']
        new_data['subj_type'] = datas[i]['subj_type']
        new_data['obj_type'] = datas[i]['obj_type']
        new_data['stanford_pos'] = datas[i]['stanford_pos']
        new_data['stanford_ner'] = datas[i]['stanford_ner']
        new_data['stanford_head'] = datas[i]['stanford_head']
        new_data['stanford_deprel'] = datas[i]['stanford_deprel']
        new_data['berkeley_head'] = datas[i]['berkeley_head']
        new_datas.append(new_data)
    with open(datafile + standford_berkeley_small , 'w') as f:
        json.dump(new_datas, f)
    with open(datafile + standford_berkeley_small+'sample' , 'w') as f:
        json.dump(new_datas[:2], f)

