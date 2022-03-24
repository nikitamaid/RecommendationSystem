from pyspark import SparkContext, SparkConf
import sys, time, random , itertools, math, json
import xgboost as xgb

def get_userFeatures(userFile):
    user_rc_rdd = userFile.map(lambda x: (x['user_id'] , x['review_count']))
    user_u_rdd = userFile.map(lambda x: (x['user_id'] , float(x['useful'])))
    user_fans_rdd = userFile.map(lambda x: (x['user_id'] , x['fans']))
    user_as_rdd = userFile.map(lambda x: (x['user_id'] , float(x['average_stars'])))
    user_funny_rdd = userFile.map(lambda x: (x['user_id'] , x['funny']))

    features_rdd = user_as_rdd.fullOuterJoin(user_u_rdd)\
        .fullOuterJoin(user_fans_rdd)\
        .mapValues(lambda x : ((0,0) if x[0] is None else x[0] + ((0 if x[1] is None else x[1]),))) \
        .fullOuterJoin(user_rc_rdd)\
        .mapValues(lambda x : ((0,0) if x[0] is None else x[0] + ((0 if x[1] is None else x[1]),))) \
        .fullOuterJoin(user_funny_rdd)\
        .mapValues(lambda x : ((0,0) if x[0] is None else x[0] + ((0 if x[1] is None else x[1]),))) 
    return features_rdd
    
def get_businessFeatures(businessFile):
    business_rdd = businessFile.map(lambda x : (x['business_id'], (x['latitude'], x['longitude'], x['review_count'] , x['is_open'])))
    business_avgStars_rdd = businessFile.map(lambda x : (x['business_id'], x['stars'])).groupByKey().mapValues(lambda x: sum(x) / len(x))

    features_rdd = business_rdd.fullOuterJoin(business_avgStars_rdd)\
                    .mapValues(lambda x : ((0,0,0) if x[0] is None else x[0] + ((0 if x[1] is None else x[1]),)))
    return features_rdd

start_time = time.time()
#input_file = 'yelp_train.csv'
#val_file_path = 'yelp_val_in.csv'
#outputFile = 'out2_1.csv'
file_path = sys.argv[1]
val_file_path = sys.argv[2]
outputFile = sys.argv[3]

conf = SparkConf().setMaster("local[*]").setAppName("task2_2")
sc = SparkContext(conf = conf)
sc.setLogLevel("ERROR")

# generating users features
userFile = sc.textFile(file_path +'user.json').map(lambda x: json.loads(x))
userFeatures_dict = get_userFeatures(userFile).collectAsMap()
print('user features done')

businessFile = sc.textFile(file_path +'business.json').map(lambda x: json.loads(x))
businessFeatures_dict = get_businessFeatures(businessFile).collectAsMap()
print('business features done')

rawRDD = sc.textFile(file_path +'yelp_train.csv').filter(lambda l: not l.startswith('user_id'))\
    .map(lambda x : (x.split(',')[0] , x.split(',')[1],  float(x.split(',')[2])))\
        .map(lambda x :(list(userFeatures_dict[x[0]]+businessFeatures_dict[x[1]]), x[2]))
print('generating users done')

X_test = sc.textFile(val_file_path +'yelp_val_in.csv').filter(lambda l: not l.startswith('user_id'))\
    .map(lambda x : (x.split(',')[0] , x.split(',')[1]))\
        .map(lambda x :(userFeatures_dict[x[0]]+businessFeatures_dict[x[1]])).collect()
print('x test generated')
X_train = rawRDD.map(lambda x: x[0]).collect()
print('x train done')
y_train = rawRDD.map(lambda x: x[1]).collect()
print('y train done')

model = xgb.XGBRegressor()
model.fit(X_train, y_train)

predict = model.predict(X_test)
predicted = list(predict)

valWriteRDD = sc.textFile(file_path +'yelp_val_in.csv').filter(lambda l: not l.startswith('user_id'))\
    .map(lambda x : (x.split(',')[0] , x.split(',')[1]))

with open(outputFile, 'w+') as writeFile:
    writeFile.write("user_id, business_id, prediction\n")
    for index, val in enumerate(predicted):
        writeFile.write(valWriteRDD[index][0] + "," + valWriteRDD[index][1] + "," + str(val) + "\n")

end_time = time.time() 
print("Duration:{0}".format(end_time - start_time))

#actualPred = sc.textFile(file_path +'yelp_val.csv').filter(lambda l: not l.startswith('user_id'))\
#    .map(lambda x :(x.split(',')[0], x.split(',')[1], float(x.split(',')[2]))).map(lambda x: x[2]).collect()
#cal = [(predicted[i] - actualPred[i]) ** 2 for i in range(len(predicted))]
#rmse = math.sqrt(sum(cal) / len(cal))
#print('RMSE {0}'.format(rmse))

