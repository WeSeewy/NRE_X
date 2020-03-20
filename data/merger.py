import json

tacred_dir = '../dataset/tacred/'
datasets = ['test', 'dev', 'train']

standford = '.json'
berkeley = '_berkeley_depend.json'
standford_berkeley = '_std_ber.json'

for ds in datasets:
    print('merge dataset ' + ds)
    datafile = tacred_dir + ds
    with open(datafile + standford, 'r') as f:
        data_st = list(json.load(f))
    with open(datafile + berkeley, 'r') as f:
        data_be = list(json.load(f))
    assert len(data_st) == len(data_be), '数据集size不一致！'
    new_datas = []

    for i in range(0, len(data_st)):
        assert len(data_st[i]['stanford_head']) == len(data_be[i]['berkeley_head']), 'json数据不对应！'
        new_data = {**(data_st[i]),**(data_be[i])}
        new_datas.append(new_data)

    # with open(datafile+ "_sample" + standford_berkeley , 'w') as f:
    #     json.dump(new_datas[0:10], f)

    with open(datafile + standford_berkeley , 'w') as f:
        json.dump(new_datas, f)