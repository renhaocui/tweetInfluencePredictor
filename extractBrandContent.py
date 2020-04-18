import json
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def extract(brandList):
    for brand in brandList:
        brandIDSet = set()
        outputFile = open(brand+'.content', 'w')
        for filename in os.listdir('adData/'+brand):
            if filename.endswith('.json'):
                inputFile = open('adData/'+brand+'/'+filename, 'r')
                for line in inputFile:
                    try:
                        data = json.loads(line.strip())
                    except Exception as e:
                        continue
                    content = data['text']
                    id = data['id']
                    if id not in brandIDSet:
                        brandIDSet.add(id)
                        if len(content) > 30:
                            outputFile.write(content.replace('\n', ' ').replace('\r', ' ')+'\n')
                inputFile.close()
        outputFile.close()



if __name__ == '__main__':
    brandList = ['Gap', 'WholeFoods']
    extract(brandList)