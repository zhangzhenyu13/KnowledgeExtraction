import collections
def countLines(filename):
    lengths=[]
    sixs=[]
    sevens=[]

    with open(filename,"r") as f:
        for line in f:
            leng=len(line.split(" "))
            lengths.append(leng)
            if leng==6:
                sixs.append(line)
            if leng==7:
                sevens.append(line)

    print(collections.Counter(lengths))
    for line in sevens:
        print(line.split(" "))

def fixFeaturesDismatchError(filename):
    '''
    some lines have 7 toks after whilte space speration
    '''
    records=[]
    with open(filename, "r") as f:
        for line in f:
            toks=line.split(" ")
            if len(toks)==6:
                records.append(line)
            else:
                del toks[2]
                toks[1]="[PAD]"
                assert len(toks)==6
                records.append(
                    " ".join(toks)
                )
    with open(filename+"-fixed", "w") as f:
        f.writelines(records)

def testScores(filename):
    from knowledgeextractor.utils.scores import scores
    #'''
    acc, precision, recall, f1 =scores(filename)
    print('%s acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' %
         (filename, acc, precision, recall, f1) )
    
    #'''
if __name__ == "__main__":
    countLines("eval/compare-test")
    fixFeaturesDismatchError("eval/compare-test")
    countLines("eval/compare-test-fixed")
    testScores("eval/compare-test-fixed")