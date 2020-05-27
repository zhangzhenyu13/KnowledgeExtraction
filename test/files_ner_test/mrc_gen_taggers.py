import json
import collections

def parseExamples(record):
    examples=[]
    #tag the words firstly
    categories=collections.defaultdict(lambda : [])
    for entity in record["entities"]:
        label=entity["label_type"]
        categories[label].append(entity)
    
    prev_ent=-1
    next_ent=-1
    for k in categories.keys():
        qas=sorted(categories[k], key=lambda x: x["start_pos"])
        for i in range(len(qas)):
            

    return examples

    
def genSquadExmaples(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        with open(file_name.replace(".json", "-mrc.json"), "w", encoding="utf-8") as f2:
            for line in f:
                if not line.strip():
                    continue
                left=line.find("{")
                right=line.rfind("}")
                line=line[left:right+1]
                record=json.loads(line)
                exmaples=parseExamples(record)
                f2.write("\n".join(map(lambda d: json.dumps(exp),exmaples))+"\n" )

if __name__ == "__main__":
    '''
    entities = [{"end_pos": 15, "label_type": "解剖部位", "start_pos": 14}, {"label_type": "解剖部位", "start_pos": 19, "end_pos": 20}, {"label_type": "解剖部位", "start_pos": 39, "end_pos": 40}, {"label_type": "解剖部位", "start_pos": 45, "end_pos": 46}, {"end_pos": 50, "label_type": "解剖部位", "start_pos": 48}, {"label_type": "手术", "start_pos": 58, "end_pos": 87}, {"label_type": "疾病和诊断", "start_pos": 94, "end_pos": 99}, {"label_type": "疾病和诊断", "start_pos": 103, "end_pos": 113}, {"label_type": "影像检查", "start_pos": 176, "end_pos": 178}, {"end_pos": 191, "label_type": "解剖部位", "start_pos": 180}, {"label_type": "手术", "start_pos": 249, "end_pos": 277}, {"end_pos": 292, "label_type": "解剖部位", "start_pos": 290}, {"end_pos": 298, "label_type": "解剖部位", "start_pos": 296}, {"label_type": "疾病和诊断", "start_pos": 302, "end_pos": 307}, {"end_pos": 327, "label_type": "药物", "start_pos": 325}, {"end_pos": 336, "label_type": "药物", "start_pos": 333}, {"end_pos": 362, "label_type": "解剖部位", "start_pos": 361}, {"end_pos": 365, "label_type": "解剖部位", "start_pos": 364}, {"end_pos": 396, "label_type": "解剖部位", "start_pos": 395}]
    entities=sorted(entities, key=lambda x: x["start_pos"])
    text="，患者2008年9月3日因“腹胀，发现腹部包块”在我院腹科行手术探查，术中见盆腹腔肿物，与肠管及子宫关系密切，遂行“全子宫左附件切除+盆腔肿物切除+右半结肠切除+DIXON术”，术后病理示颗粒细胞瘤，诊断为颗粒细胞瘤IIIC期，术后自2008年11月起行BEP方案化疗共4程，末次化疗时间为2009年3月26日。之后患者定期复查，2015-6-1，复查CT示：髂嵴水平上腹部L5腰椎前见软组织肿块，大小约30MM×45MM，密度欠均匀，边界尚清楚，轻度强化。查肿瘤标志物均正常。于2015-7-6行剖腹探查+膀胱旁肿物切除+骶前肿物切除+肠表面肿物切除术，术程顺利，，术后病理示：膀胱旁肿物及骶前肿物符合颗粒细胞瘤。于2015-7-13、8-14给予泰素240MG+伯尔定600MG化疗2程，过程顺利。出院至今，无发热，无腹痛、腹胀，有脱发，现返院复诊，拟行再次化疗收入院。起病以来，精神、胃纳、睡眠可，大小便正常，体重无明显改变。"
    
    ExampleSementer(32, key_text="text").seg_single_exmple({"text":text, "entities":entities})
    
    exit(-1)
    #'''
    
    max_seq_length=384
    source_file="/home/zhangzy/nlpdata/MRC/data.json"

    genSquadExmaples(source_file)

    
