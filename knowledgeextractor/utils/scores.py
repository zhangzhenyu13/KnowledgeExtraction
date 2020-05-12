import subprocess
import os

def scores(path):
    print("***********subprocess score, from",path, os.path.exists(path))
    bashCommand = 'perl /home/zhangzy/KnowledgeExtraction/knowledgeextractor/utils/conlleval'
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE,
                               stdin=open(path))
    output, error = process.communicate()
    #print("******output*********\n",output)
    output = output.decode().split('\n')[1].split('%; ')
    output = [out.split(' ')[-1] for out in output]
    acc, prec, recall, fb1 = tuple(output)
    return float(acc), float(prec), float(recall), float(fb1)