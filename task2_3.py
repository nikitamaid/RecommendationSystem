from pyspark import SparkContext, SparkConf
import sys, time, random , itertools, math, json
import xgboost as xgb

conf = SparkConf().setAppName('task2_3').setMaster('local')
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

def model_based(file_path , val_file_path):

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

    output = list()
    for i in range(len(predicted)):
        output.append((valWriteRDD[i][0], valWriteRDD[i][1], predicted[i]))
    
    return output



def item_based(input_file , val_file_path):

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
    user_rating_rdd = rawRDD.map(lambda x:((x[0],x[2]))).collectAsMap()
    user_ratingNum_rdd = user_rating_rdd.groupByKey().mapValues(list).mapValues(lambda x : len(x))
    user_ratingNum_dict = user_ratingNum_rdd.collectAsMap()
    max_num_rating = max(list(user_ratingNum_dict.values()))
    #print("Max num rating " + str(max_num_rating))

    user_num_ratings_rdd = user_ratingNum_rdd.map(lambda x : (x[0], x[1]/max_num_rating))
    #print(user_num_ratings_rdd.take(10))

    weights = user_num_ratings_rdd.collectAsMap()
    # print(businessUser_RatingDict)
    output = valRDD.map(lambda x:(x[0] , x[1] ,predict_ratings(x[0],x[1]))).collect()

    return output,weights
    
def read_csv(filepath):
    input_rdd =  sc.textFile(filepath)
    header = input_rdd.first()
    input_rdd = input_rdd.filter(lambda row : row != header)   #filter out header
    return input_rdd
def calculate_rmse(test_user_business_prediction):
    actual_ratings_rdd = read_csv('../resource/lib/publicdata/' + 'yelp_test_ans.csv')
    user_business_rating_rdd = actual_ratings_rdd.map(lambda x: (x.split(',')[0], x.split(',')[1], float(x.split(',')[2])))
    actual_ratings = user_business_rating_rdd.map(lambda x : x[2]).collect()
    predicted_ratings = test_user_business_prediction

    x = [(predicted_ratings[i][2] - actual_ratings[i]) ** 2 for i in range(len(predicted_ratings))]
    rmse = math.sqrt(sum(x) / len(x))

    return rmse


def main():
    start_time = time.time()
    input_file = sys.argv[1]
    val_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    model_pred = model_based(input_file , val_file_path)
    item_pred, weigths = item_based(input_file , val_file_path)
    res = []
    for i in range(len(item_pred)):
        user = item_pred[i][0]
        business = item_pred[i][1]

        itemRating = item_pred[i][2]
        modelRating = model_pred[i][2]

        w = weigths[user]

        res.append((user, business, (w)*(itemRating) + (1-w)*(modelRating)))


    output = []
    for i in range(len(res)):
        output.append(res[i])


    with open(output_file_path, 'w+') as f:
        f.write("user_id, business_id, prediction\n")
        for i in output:
            f.write(i[0] + "," + i[1] + "," + str(i[2]) + "\n")

    end_time = time.time()
    print("Duration: {0}".format(end_time-start_time))
    rmse = calculate_rmse(hybrid_predictions)
    print(rmse)