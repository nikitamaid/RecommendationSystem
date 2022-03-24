from pyspark import SparkContext, SparkConf
import sys, time, random , itertools, math

THRESHOLD_USERS = 10
WEIGHT = 0.6
NEIGHBOURS = 125


def get_business(userid):
    user_dict = userBusiness_dict[userid]
    businesses = user_dict.keys()
    ratings = user_dict.values()
    return list(businesses),list(ratings)

def get_users(business):
    if business in businessUser_dict.keys():
        return businessUser_dict[business]
    elif business not in businessUser_dict.keys():
        return businessUser_dict.get(business, {})


def pearson_sim(businessid, b):
    users1 = get_users(businessid)
    user_1 = users1.keys()
    users2 = get_users(b)
    users_2 = users2.keys()
    
    common_users = set(user_1).intersection(set(users_2))
    if(len(common_users) < THRESHOLD_USERS):
        return WEIGHT

    r1 = []
    r2 = []
    for cu in common_users:
        r1.append(users1[cu])
        r2.append(users2[cu])    
    
    avg_r1 = sum(r1)/len(r1)
    avg_r2 = sum(r2)/len(r2)
     
    new_r1 = [r - avg_r1 for r in r1]
    new_r2 = [r - avg_r2 for r in r2]

    num = 0
    deno = 0
    d1 = 0
    d2 = 0

    for i in range(len(new_r1)):
        num += new_r1[i]*new_r2[i]
    d1 = sum([r**2 for r in new_r1])
    d2 = sum([r**2 for r in new_r2])

    deno = math.sqrt(d1)*math.sqrt(d2)

    if deno != 0 :
        return num/deno
    
    return 0
    
def get_rating(pairs):
    num = 0
    deno = 0
    num = sum([i[0]*i[1] for i in pairs])
    deno = sum(abs(i[0]) for i in pairs)
    return num/deno


def predict_ratings(userid,businessid):
    weights = list()
    businesses, ratings = get_business(userid)
    for b in businesses:
        weights.append(pearson_sim(businessid, b))
    # print(weights)
    # print(ratings)
    pair = list(zip(weights,ratings))
    pairs = sorted(pair , key = lambda x: -abs(x[0]))
    paired = pairs[0:min(NEIGHBOURS, len(pairs))]
    rating = get_rating(paired)

    return rating

start_time = time.time()
#input_file = 'yelp_train.csv'
#val_file_path = 'yelp_val_in.csv'
#outputFile = 'out2_1.csv'
input_file = sys.argv[1]
val_file_path = sys.argv[2] 
outputFile = sys.argv[3]

conf = SparkConf().setMaster("local[*]").setAppName("task2_1")
sc = SparkContext(conf = conf)
sc.setLogLevel("ERROR")

rawRDD  = sc.textFile(input_file).filter(lambda l: not l.startswith('user_id'))\
    .map(lambda x : (x.split(',')[0] , x.split(',')[1],  float(x.split(',')[2])))

valRDD  = sc.textFile(val_file_path).filter(lambda l: not l.startswith('user_id'))\
    .map(lambda x : (x.split(',')[0] , x.split(',')[1]))    

userBusiness_dict = rawRDD.map(lambda x:(x[0],(x[1], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
# print(userBusiness_dict)
# print(len(userBusiness_dict))
businessUser_dict = rawRDD.map(lambda x:(x[1],(x[0],float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
# print(businessUser_dict)
# print(len(businessUser_dict))
businessUser_RatingDict = rawRDD.map(lambda x:((x[1],x[0]),float(x[2]))).collectAsMap()
# print(businessUser_RatingDict)
output = valRDD.map(lambda x:(x[0] , x[1] ,predict_ratings(x[0],x[1]))).collect()

with open(outputFile , 'w') as outFile:
    outFile.write('user_id, business_id, prediction\n')
    for i in output:
        outFile.write(i[0]+','+i[1]+','+str(i[2])+'\n')

end_time = time.time()
print("Duration: ",(end_time-start_time))

# reference_file=open('yelp_val.csv',"r")
# output_file=open(outputFile,"r")

# n=0
# rmse=0


# while(True):
#     l1=output_file.readline()
#     l2=reference_file.readline()
#     if "user_id" in l2:
#         continue
#     if l2=="":
#         break
#     n+=1
#     rmse+=math.pow(float(l1.split(",")[2][:-1])-float(l2.split(",")[2][:-1]),2)
#     if not l1 and not l2:
#         break

# rmse=math.sqrt(rmse/n)

# print(rmse)