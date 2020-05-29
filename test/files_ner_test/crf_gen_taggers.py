import os
import json
import collections
#from textblob import TextBlob
from knowledgeextractor.utils.text_segment import to_sentences

class ExampleSementer:
    def __init__(self, max_seq_length, key_text="originalText", key_entities="entities"):
        self.max_seq_length=max_seq_length
        self.initState()
        self.key_entities=key_entities
        self.key_text=key_text
        
    def initState(self):
        self._curLen=0
        self._offset=0
        self._pos=0

        self._txts=[]

    @staticmethod
    def print_record(record):
        txt, ents= record["text"], record["entities"]
        print(txt)
        print(ents)
        print(record)
        for ent in ents:
            s,e=ent["start_pos"], ent["end_pos"]
            print(ent, txt[s:e])
        print("*"*50,"\n")

    def _get_record(self, entities):
        ents=[]
        txts=self._txts
        #print(self._pos, self._curLen, self._offset, len(entities))
        while self._pos<len(entities) and entities[self._pos]["end_pos"]<self._offset+self._curLen:
                ent=entities[self._pos]
                ent["start_pos"]-=self._offset
                ent["end_pos"]-=self._offset
                ents.append(ent)
                self._pos+=1
        self._offset+=self._curLen
        self._curLen=0
        txt="".join(txts)
        record={"text":txt, "entities":ents}
        self._txts.clear()
        return record
    
    def seg_single_exmple(self, record):
        # senetence centered segmentation
        # split sentences into records with 
        # each record containing several sents and coresponding entities
        entities=record[self.key_entities]
        text=record[self.key_text]
        sents=[]
        self.initState()

        records=[]

        lengths=[]
        length_pos=0

        sentences=  to_sentences(text)

        for sent in sentences:
            #print(sent)
            sents.append(sent)
            lengths.append([length_pos,len(sent)])
            length_pos+=len(sent)
        #print(lengths)

        for sent in sents:
            if self._curLen+len(sent)>max_seq_length and len(self._txts)>0:
                record=self._get_record(entities)
                records.append(record)
            self._curLen+=len(sent)
            self._txts.append(sent)
                

        if self._curLen>0:
            record=self._get_record(entities)
            records.append(record)
        
        #for record in records:
        #    ExampleSementer.print_record(record)
        #print("\n", "-"*50, "\n")
        
        return records

    def gatherLabels(self, records):
        labels=set()
        for record in records:
            token_labels= record["token_labels"]
            labels=labels.union(token_labels)
                
        if "O" not in labels:
            labels.add("O")
        
        labels=sorted(labels)
        return labels

    def label_single_example(self, record):
        # label the example and then seg it into several records
        entities=record["entities"]
        text=record["text"]
        sequence=[[word, "O"] for word in text]
        nested=0
        labels=set()

        for entity in entities:
            s, e= entity["start_pos"], entity["end_pos"]
            label=entity["label_type"]      
            labels.add(label+"-B")
            labels.add(label+"-I")      
            for i in range(s, e):
                if sequence[i][0][-1]=="I":
                    nested+=1

                sequence[i]=(sequence[i][0], label+"-I")
                if i==s:
                    sequence[i]=(sequence[i][0], label+"-B")
        
        record={
                "text": "".join([w for w,_ in sequence]),
                "token_labels": [t for _,t in sequence]
        }
        
        return record

def segCMIDRecords(file_name,max_seq_len):
    lengths=[]
    segger=ExampleSementer(max_seq_len)
    records=[]

    with open(file_name, "r", encoding="utf-8") as f:
        
        with open(file_name.replace(".json", "-clean.json"), "w", encoding="utf-8") as f2:
            for line in f:
                if not line.strip():
                    continue
                left=line.find("{")
                right=line.rfind("}")
                line=line[left:right+1]
                
                records.extend( segger.seg_single_exmple(json.loads(line)) )
                
            for record in records:
                if record["entities"]:
                    
                    obj={"originalText":record["text"], "entities":record["entities"]}
                    lengths.append(len(obj["originalText"]))
                    #print(obj)
                    f2.write(json.dumps(obj,ensure_ascii=False)+"\n")
        print(sorted(dict(collections.Counter(lengths)).items(),key=lambda x:x[0] ) )


def fromCMIDtoCONLLFile(file_name, folder):
    count=0
    null=0
    nested=0
    labeled_tokens=[]
    labels=[]

    with open(file_name, "r") as f:
        for line in f:
            if not line.strip():
                continue
            
            record=json.loads(line)
            #print(line)
            #record=json.loads(line)
            #print(record["entities"])
            if len(record["entities"])>0:
                count+=1
                
                sequence=[(w, "O") for w in record["originalText"]]
                for entity in record["entities"]:
                    
                    s, e= entity["start_pos"], entity["end_pos"]
                    label=entity["label_type"]
                    labels.append(label+"-I")
                    labels.append(label+"-B")
                    
                    for i in range(s, e):
                        if sequence[i][0][-1]=="I":
                            nested+=1

                        sequence[i]=(sequence[i][0], label+"-I")
                        if i==s:
                            sequence[i]=(sequence[i][0], label+"-B")

                labeled_tokens.append(sequence)
            else:
                null+=1
            

    print(collections.Counter(labels))    
    print(count, null, nested)
    
    def sequence2labeledtags(sequences, file_name2):
        result=[]
        for sequence in sequences:
            for token, label in sequence:
                result.append("{} {}\n".format(token, label))
            result.append("\n")

        with open(file_name2, "w") as f:
            f.writelines(result)

    dev_size=test_size= int(0.05*len(labeled_tokens))
    train, dev, test= os.path.join(folder,"train.txt"),os.path.join(folder,"dev.txt"),os.path.join(folder,"test.txt")

    sequence2labeledtags(labeled_tokens[:-(dev_size+test_size)], train)
    sequence2labeledtags(labeled_tokens[-(dev_size+test_size):-test_size], dev)
    sequence2labeledtags(labeled_tokens[-test_size:], test)

    
    file_label=os.path.join(folder,"labels.txt")

    labels=list(collections.Counter(labels).keys())+["O"]

    with open(file_label, "w") as f:
        f.writelines("\n".join(labels))


def taggerMapping(tagger_map_file, records_file):
    with open(tagger_map_file, "r", encoding="utf-8") as f:
        tag_mapper=json.load(f)

    records=[]
    with open(records_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            left=line.find("{")
            right=line.rfind("}")
            line=line[left:right+1]
            record=json.loads(line)
            for entity in record["entities"]:
                entity["label_type"]=tag_mapper[entity["label_type"]]
            
            records.append(record)

    with open(records_file.replace(".json", "-mapped.json"), "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record)+"\n")
    
def genTaggerFile(file_name):
    lengths=[]
    labels=set()
    records=[]
    segger=ExampleSementer(max_seq_length)
    with open(file_name, "r", encoding="utf-8") as f:
        with open(file_name.replace(".json", "-seg.json"), "w", encoding="utf-8") as f2:
            for line in f:
                if not line.strip():
                    continue
                left=line.find("{")
                right=line.rfind("}")
                line=line[left:right+1]
                
                records.extend( map(lambda record: segger.label_single_example(record) ,
                    segger.seg_single_exmple(json.loads(line)) )
                )

            for record in records:
                #print(record)
                tags=record["token_labels"]
                words=record["text"]
                if len(words)<min_sequence_length:
                    continue
                assert len(tags)==len(words)
                lengths.append(len(tags))
                f2.write(json.dumps(record,ensure_ascii=False)+"\n")

        labels=segger.gatherLabels(records)
        print(sorted(dict(collections.Counter(lengths)).items(),key=lambda x:x[0] ) )
        print(labels)

        with open(file_name.replace(".json", ".label"), "w", encoding="utf-8") as f:
            f.write("\n".join(labels)+"\n")

if __name__ == "__main__":
    '''
    entities = [{"end_pos": 15, "label_type": "解剖部位", "start_pos": 14}, {"label_type": "解剖部位", "start_pos": 19, "end_pos": 20}, {"label_type": "解剖部位", "start_pos": 39, "end_pos": 40}, {"label_type": "解剖部位", "start_pos": 45, "end_pos": 46}, {"end_pos": 50, "label_type": "解剖部位", "start_pos": 48}, {"label_type": "手术", "start_pos": 58, "end_pos": 87}, {"label_type": "疾病和诊断", "start_pos": 94, "end_pos": 99}, {"label_type": "疾病和诊断", "start_pos": 103, "end_pos": 113}, {"label_type": "影像检查", "start_pos": 176, "end_pos": 178}, {"end_pos": 191, "label_type": "解剖部位", "start_pos": 180}, {"label_type": "手术", "start_pos": 249, "end_pos": 277}, {"end_pos": 292, "label_type": "解剖部位", "start_pos": 290}, {"end_pos": 298, "label_type": "解剖部位", "start_pos": 296}, {"label_type": "疾病和诊断", "start_pos": 302, "end_pos": 307}, {"end_pos": 327, "label_type": "药物", "start_pos": 325}, {"end_pos": 336, "label_type": "药物", "start_pos": 333}, {"end_pos": 362, "label_type": "解剖部位", "start_pos": 361}, {"end_pos": 365, "label_type": "解剖部位", "start_pos": 364}, {"end_pos": 396, "label_type": "解剖部位", "start_pos": 395}]
    entities=sorted(entities, key=lambda x: x["start_pos"])
    text="，患者2008年9月3日因“腹胀，发现腹部包块”在我院腹科行手术探查，术中见盆腹腔肿物，与肠管及子宫关系密切，遂行“全子宫左附件切除+盆腔肿物切除+右半结肠切除+DIXON术”，术后病理示颗粒细胞瘤，诊断为颗粒细胞瘤IIIC期，术后自2008年11月起行BEP方案化疗共4程，末次化疗时间为2009年3月26日。之后患者定期复查，2015-6-1，复查CT示：髂嵴水平上腹部L5腰椎前见软组织肿块，大小约30MM×45MM，密度欠均匀，边界尚清楚，轻度强化。查肿瘤标志物均正常。于2015-7-6行剖腹探查+膀胱旁肿物切除+骶前肿物切除+肠表面肿物切除术，术程顺利，，术后病理示：膀胱旁肿物及骶前肿物符合颗粒细胞瘤。于2015-7-13、8-14给予泰素240MG+伯尔定600MG化疗2程，过程顺利。出院至今，无发热，无腹痛、腹胀，有脱发，现返院复诊，拟行再次化疗收入院。起病以来，精神、胃纳、睡眠可，大小便正常，体重无明显改变。"
    
    ExampleSementer(32, key_text="text").seg_single_exmple({"text":text, "entities":entities})
    
    exit(-1)
    #'''

    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/zhangzy/nlpdata/CRF/data.json")
    parser.add_argument("--mapper", default="/home/zhangzy/KnowledgeExtraction/config/label_map.json")

    parser.add_argument("--max_sequence_length", default=382)
    parser.add_argument("--min_sequence_length", default=32)


    args=parser.parse_args()

    source_file=args.data
    tagger_map_file=args.mapper
    max_seq_length=args.max_sequence_length
    min_sequence_length=args.min_sequence_length

    taggerMapping(tagger_map_file,source_file)
    genTaggerFile(source_file.replace(".json", "-mapped.json"))

    #result_folder="/home/zhangzy/nlpdata/CRF"
    #cleamCMID(source_file,max_seq_length)
    
    #source_file="/home/zhangzy/KnowledgeExtraction/data/ner/yidu7k/ner-clean.json"
    #CMID2CONLLFile(source_file, result_folder)
    
