from pyspark import SparkContext, SparkConf
import sys, time, random , itertools


SIMILARITY =0.5
BANDS = 50
ROWS = 5

def computing_min(r1,r2):
    temp = list()
    for i,j in zip(r1,r2):
        temp.append(min(i,j))
    return temp


def hash_function(x):
    hashFuncs = list()
    temp = list()
    numFuncs = BANDS*ROWS
    hashFuncs.append(x)
    for i in range(1,numFuncs+1):
        a = random.randint(1, sys.maxsize-1)
        b = random.randint(1, sys.maxsize -1)
        p = 21842137
        func = (((a*x)+b)%p)%num_of_users
        temp.append(func)
    hashFuncs.append(temp)
    return hashFuncs

def matrixSpliting(x):
    split = list()
    for i,j in enumerate(range(1,len(x)+1,5)):
        split.append((i-1, hash(tuple(x[j-1:j+4]))))
    return split

def jaccard_similarity(candidates,businessUserDict,IndexBusinessDict):
    output = list()
    computed = set()
    for i in candidates:
        if i not in computed:
            computed.add(i)
            inter = float(len(set(businessUserDict[i[1]]).intersection(set(businessUserDict[i[0]]))))
            union = float(float(len(set(businessUserDict[i[0]])))+float(len(set(businessUserDict[i[1]])))-inter)
            js = inter/union
            if js >= SIMILARITY:
                if IndexBusinessDict[i[0]] < IndexBusinessDict[i[1]]:
                    output.append([IndexBusinessDict[i[0]],IndexBusinessDict[i[1]],js])
                elif IndexBusinessDict[i[1]] < IndexBusinessDict[i[0]]:
                    output.append([IndexBusinessDict[i[1]],IndexBusinessDict[i[0]],js])
    output.sort(key = lambda x: (x[0],x[1]))
    return output


 


start_time = time.time()
input_file = sys.argv[1]
outputFile = sys.argv[2]

conf = SparkConf().setMaster("local[*]").setAppName("task1")
sc = SparkContext(conf = conf)
sc.setLogLevel("ERROR")

rawRDD  = sc.textFile(input_file).filter(lambda l: not l.startswith('user_id'))\
    .map(lambda x : (x.split(',')[0] , x.split(',')[1]))
usersIndexRDD = rawRDD.map(lambda x: x[0]).distinct().zipWithIndex()
usersDict = usersIndexRDD.collectAsMap()
# print(usersDict)
num_of_users = usersIndexRDD.count()

businessIndexDict = rawRDD.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
# print(businessIndexDict)
IndexBusinessDict = {v:k for k,v in businessIndexDict.items()}
# print(IndexBusinessDict)


usersMapping = usersIndexRDD.mapValues(hash_function).map(lambda x:(x[1])).map(lambda x:(x[0],x[1]))
# print(usersMapping.collect())

userBusinessDict = rawRDD.map(lambda x: [usersDict[x[0]],businessIndexDict[x[1]]]).groupByKey().mapValues(list)
# print(user_business)
businessUserDict = rawRDD.map(lambda x: [businessIndexDict[x[1]],usersDict[x[0]]]).groupByKey().mapValues(list).collectAsMap()
# print(business_user)

signatureMatrix = userBusinessDict.leftOuterJoin(usersMapping).map(lambda x:x[1])\
                    .flatMap(lambda x: [(i, x[1]) for i in x[0]]).reduceByKey(computing_min)
# print(signatureMatrix.collect())
candidatePairs = signatureMatrix.flatMap(lambda x: [(tuple(i), x[0]) for i in matrixSpliting(x[1])])\
                .groupByKey().map(lambda x: list(x[1])).filter(lambda x: len(x) >1)\
                .flatMap(lambda x: [i for i in itertools.combinations(x, 2)]).collect()
output = jaccard_similarity(set(candidatePairs) , businessUserDict , IndexBusinessDict)


with open(outputFile , 'w') as writeFile:
    writeFile.write('business_id_1, business_id_2, similarity'+'\n')
    for i in output:
        writeFile.write(i[0]+','+i[1]+','+str(i[2])+'\n')
end_time = time.time()
print("Duration:",end_time-start_time)
