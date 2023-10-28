import os
import csv

path = './dataset_qb/'
for files in os.listdir(path):
    datas = []
    # 读取csv文件
    for fname in os.listdir(path):
        if 'csv' in fname:
            fname_path = path + fname
            with open(fname_path, "r") as csvfile:
                reader = csv.reader(csvfile)
                # 去掉表头（第一行）
                reader = list(reader)[1:]
                for line in reader:
                    datas.append(line)
excel_name = './dataset_qb/qb_lxb_250000.csv'
csv_head = [
    'damageLevel',
    'hurt',
    'collapse',
    'deltaP',
    'ammoQuantity',
    'outerSideLength',
    'outerSideWidth',
    'outerHeight',
    'roofThick',
    'wallThick',
    'soilThick',
    'liningPlateThick',
]

with open(excel_name, 'w') as csvfile2:
    writer = csv.writer(csvfile2)
    # 写表头
    writer.writerow(csv_head)
    writer.writerows(datas)

print
'finish~'
