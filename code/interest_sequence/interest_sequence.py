import data
from collections import defaultdict
from itertools import combinations, product
import numpy as np
from scipy.stats import pearsonr
import math
import time
import pickle

def generate_seq(df, users):
	# key:user val: (item, rating) in sorted order
	seq = defaultdict(list)
	# key:user val: list of ratings
	ratings = defaultdict(list)
	# key:user val: list of items rated
	items = defaultdict(list)
	# dict of dict with {key:user {key:item val: rating}}
	item_ratings = {}

	for u in users:
		df_u = df[df['userId'] == u]
		df_u = df_u.sort_values(by = ['timestamp'])
		items[u].append(df_u['movieId'].unique())
		item_ratings[u] = {}
		for index, row in df_u.iterrows():
			seq[u].append((row['movieId'], row['rating']))
			ratings[u].append(row['rating'])
			item_ratings[u][row['movieId']] = row['rating']

	return seq, ratings, items, item_ratings

def deviation_ratings(r_u, r_v, r_u_i, r_v_i):
	max_ru, min_ru, max_rv, min_rv = np.max(r_u), np.min(r_u), np.max(r_v), np.min(r_v)
	if (min_ru == max_ru) or (min_rv == max_rv):
		return 1
	return np.abs((r_u_i - max_ru)/(max_ru - min_ru) - (r_v_i - max_rv)/(max_rv - min_rv))

def compute_lcsis(seq, users, ratings, items, theta = 0.2):
	# key: (user u,user v) val: |lcsis(u,v)|
	lcsis = {}
	# key: (user u,user v) val: no. of items rated by both u and v [used to calculate similarity]
	common_count = {}
	# key: (user u,user v) val: items rated by both u and v [used to calculate similarity]
	common_items = {}
	# key: (user u,user v) val: no. of items rated by u + no. of items rated by v [used to calculate similarity]
	total = {}
	length = len(users)
	length = length * (length - 1) * 0.5
	comb = 0
	for u, v in combinations(users, 2):
		if comb % 100 == 0:
			print('[', comb, '/', length, '] complete\r', end = '')
		comb += 1

		m, n = len(seq[u]), len(seq[v])
		w = np.zeros((m+1, n+1))

		for i in range(m+1):
			for j in range(n+1):
				if(i ==0 or j==0):
					w[i][j] = 0
				else:
					dev = deviation_ratings(ratings[u], ratings[v], ratings[u][i-1], ratings[v][j-1])
					if(seq[u][i-1][0] == seq[v][j-1][0] and dev <= theta):
						w[i][j] = w[i-1][j-1] + 1
					else:
						w[i][j] = max(w[i,j-1], w[i-1,j])

		lcsis[(u,v)] = w[m][n]
		total[(u,v)] = m + n

		if(len(items[u][0])>len(items[v][0])):
			common_items[(u,v)] = list(set(items[u][0]).intersection(items[v][0]))
			common_count[(u,v)] = len(common_items[(u,v)])
		else:
			common_items[(u,v)] = list(set(items[v][0]).intersection(items[u][0]))
			common_count[(u,v)] = len(common_items[(u,v)])

	for u in users:
		lcsis[(u,u)] = len(seq[u])

	return lcsis, total, common_count, common_items

def compute_position(j, u, v, i, tupple_i, seq_v, ratings, theta):
	target_item = tupple_i[0]
	for x in range(1, j+1):
		if(seq_v[x-1][0] == target_item):
			dev = deviation_ratings(ratings[u], ratings[v], ratings[u][i-1], ratings[v][x-1])
			if(dev <= theta):
				return x
	return 0

def compute_acsis(seq, users, ratings, theta = 0.2):

	# key: (user u,user v) val: |acsis(u,v)|
	acsis = {}
	length = len(users)
	length = length * (length - 1) * 0.5
	comb = 0
	for u, v in combinations(users, 2):
		if comb % 100 == 0:
			print('[', comb, '/', length, '] complete\r', end = '')
		comb += 1
		m, n = len(seq[u]), len(seq[v])
		w = np.zeros((m+1, n+1))

		for i in range(m+1):
			for j in range(n+1):
				if(i ==0 or j==0):
					w[i][j] = 1
				else:
					x = compute_position(j, u, v, i, seq[u][i-1], seq[v], ratings, theta)
					if(x == 0):
						w[i][j] = w[i-1][j]
					else:
						w[i][j] = w[i-1][j] + w[i-1][x-1]

		acsis[(u,v)] = w[m][n]

	for u in users:
		acsis[(u,u)] = 2**len(seq[u])
	return acsis

def similarity_is(lcsis, acsis, users, total, common, alpha = 1):
	length = len(users)
	length = length * (length - 1) * 0.5
	comb = 0
	sim = {}
	# key: (user u,user v) val: similarity based on interest sequence
	weighted_sim = {}
	for u,v in combinations(users, 2):
		comb += 1
		if(u!=v):
			sim_lcsis = lcsis[(u,v)] / (math.sqrt(lcsis[(u,u)] * lcsis[(v,v)]))
			sim_acsis = acsis[(u,v)] / (math.sqrt(acsis[(u,u)]) * math.sqrt(acsis[(v,v)]))
			sim[(u,v)] = alpha * np.exp(sim_lcsis) + (1-alpha) * np.exp(sim_acsis)			 
			weighted_sim[(u,v)] = (sim[(u,v)] * common[(u,v)]) / total[(u,v)]

	return sim, weighted_sim

def similarity(weighted_sim, users, seq, common_items, item_ratings):
	length = len(users)
	length = length * (length - 1) * 0.5
	comb = 0
	# key: (user u,user v) val: similarity based on interest sequence * pearson correlaiton
	similarity_list = defaultdict(list)
	for u,v in combinations(users, 2):
		comb += 1
		if(len(common_items[(u,v)])<=1):
			sim = 0
		else:
			common_rating_u = []
			common_rating_v = []

			for item in common_items[(u,v)]:
				common_rating_u.append(item_ratings[u][item])
				common_rating_v.append(item_ratings[v][item])

			corr,_ = pearsonr(common_rating_u, common_rating_v)
			if(np.isnan(corr)):
				sim = 0
			else:
				sim = corr * weighted_sim[(u,v)]
		similarity_list[u].append((v, sim))
		similarity_list[v].append((u, sim))

	return similarity_list

def predict(u, i, similarity_list, K, ratings, item_ratings):
	r_hat = np.mean(ratings[u]) 
	sorted_sim_users = sorted(similarity_list[u], key = lambda x:x[1], reverse=True)

	num = 0 
	deno = 0
	count = 0
	for t in sorted_sim_users:
		v = t[0]
		sim = t[1]
		# if user v has rated item i
		if(i in item_ratings[v]):
			num += sim * (item_ratings[v][i] - np.mean(ratings[v]))
			deno += sim
			count += 1
			if(count == K):
				break
	
	if(deno != 0):	
		r_hat += num/deno

	return r_hat

def rating_prediction(df_test, ratings, item_ratings, similarity_list, K):
	count = 0
	sq_sum = 0
	abs_sum = 0
	test_set = []
	for index, row in df_test.iterrows():
		r_hat = predict(row['userId'], row['movieId'], similarity_list, K, ratings, item_ratings)
		test_set.append((row['userId'], row['movieId'], row['rating'], r_hat, {}))
		sq_sum += (r_hat - row['rating']) ** 2
		abs_sum += abs(r_hat - row['rating'])
		count += 1
	
	return test_set, math.sqrt(sq_sum / count), abs_sum / count

def precision_recall_calculation(predictions, threshold=3.5):

    # First map the predictions to each user.
    user_predict_true = defaultdict(list)
    for user_id, movie_id, true_rating, predicted_rating, _ in predictions:
        user_predict_true[user_id].append((predicted_rating, true_rating))

    precisions = dict()
    recalls = dict()
    for user_id, user_ratings in user_predict_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        no_of_relevant_items = sum((true_rating >= threshold) for (predicted_rating, true_rating) in user_ratings)

        # Number of recommended items in top 10
        no_of_recommended_items = sum((predicted_rating >= threshold) for (predicted_rating, true_rating) in user_ratings[:10])

        # Number of relevant and recommended items in top 10
        no_of_relevant_and_recommended_items = sum(((true_rating >= threshold) and (predicted_rating >= threshold)) for (predicted_rating, true_rating) in user_ratings[:10])

        # Precision: Proportion of recommended items that are relevant
        precisions[user_id] = no_of_relevant_and_recommended_items / no_of_recommended_items if no_of_recommended_items != 0 else 1

        # Recall: Proportion of relevant items that are recommended
        recalls[user_id] = no_of_relevant_and_recommended_items / no_of_relevant_items if no_of_relevant_items != 0 else 1

    # Averaging the values for all users
    average_precision=sum(precision for precision in precisions.values()) / len(precisions)
    average_recall=sum(recall for recall in recalls.values()) / len(recalls)
    F_score=(2*average_precision*average_recall) / (average_precision + average_recall)
    
    return [average_precision, average_recall, F_score]

def interest_sequence(df_train, df_test, mode, alpha, K):
	start_time = time.time()
	users = df_train['userId'].unique()

	if(mode == "Train"):
		# generating sequence of each user
		seq, ratings, items, item_ratings = generate_seq(df_train, users)
		
		# compute lcsis values between every pair of users
		print("\nComputing LCSIS")
		lcsis, total, common_count, common_items = compute_lcsis(seq, users, ratings, items)

		# storing the results for use in the future
		file = open('store/lcsis.txt', 'wb')
		pickle.dump(lcsis, file)
		file.close()
		file = open('store/total.txt', 'wb')
		pickle.dump(total, file)
		file.close()
		file = open('store/common_count.txt', 'wb')
		pickle.dump(common_count, file)
		file.close()
		file = open('store/common_items.txt', 'wb')
		pickle.dump(common_items, file)
		file.close()
		print("Elapsed Time: ", time.time() - start_time)

		# computing acsis values between every pair of users
		print("\nComputing ACSIS")
		acsis = compute_acsis(seq, users, ratings)
		# storing the array
		file = open('store/acsis.txt', 'wb')
		pickle.dump(acsis, file)
		file.close()
		print("Elapsed Time: ", time.time() - start_time)

		# computing similiarity between every pair of users based on lcsis and ascsis
		print("\nComputing Similaritiy Weights")
		sim_is, weighted_sim = similarity_is(lcsis, acsis, users, total, common_count, alpha)
		file = open('store/sim_is.txt', 'wb')
		pickle.dump(sim_is, file)
		file.close()
		file = open('store/weighted_sim.txt', 'wb')
		pickle.dump(weighted_sim, file)
		file.close()
		print("Elapsed Time: ", time.time() - start_time)

		# computing final user similarity based on pearson correlation and similarity based on IS
		print("\nComputing User Similarities")
		similarity_list = similarity(weighted_sim, users, seq, common_items, item_ratings)
		file = open('store/similarity_list.txt', 'wb')
		pickle.dump(similarity_list, file)
		file.close()
		print("Elapsed Time: ", time.time() - start_time)
		
		# Prediction new ratings and computing error
		print("\nPredicting Ratings:")
		predictions, rmse, mae = rating_prediction(df_test, ratings, item_ratings, similarity_list, K)
		file = open('store/predictions.txt', 'wb')
		pickle.dump(predictions, file)
		file.close()
		file = open('store/rmse.txt', 'wb')
		pickle.dump(rmse, file)
		file.close()
		file = open('store/mae.txt', 'wb')
		pickle.dump(mae, file)
		file.close()
		print("Elapsed Time: ", time.time() - start_time)
		
		# print("\nPredicted Ratings for a few users:")
		# print(predictions[:10])

		[precision, recall, F_score] = precision_recall_calculation(predictions, threshold=3.5)
		print("\n" + "-"*50)
		print("alpha = ", alpha, "K = ", K)
		print("RMSE:", rmse)
		print("MAE:", mae)
		print("Precision: ", precision)
		print("Recall: ", recall)
		print("F-Score: ",F_score)
		print("-"*50)

		print("TIME TAKEN FOR ENTIRE COMPUTATION: ", time.time()- start_time)


	if(mode == "Test"):
		seq, ratings, items, item_ratings = generate_seq(df_train, users)

		# read the files from storage
		file = open('store/lcsis.txt', 'rb')
		lcsis = pickle.load(file)
		file.close()
		file = open('store/total.txt', 'rb')
		total = pickle.load(file)
		file.close()
		file = open('store/common_count.txt', 'rb')
		common_count = pickle.load(file)
		file.close()
		file = open('store/common_items.txt', 'rb')
		common_items = pickle.load(file)
		file.close()
		file = open('store/acsis.txt', 'rb')
		acsis = pickle.load(file)
		file.close()

		print("\nComputing Similaritiy Weights")
		sim_is, weighted_sim = similarity_is(lcsis, acsis, users, total, common_count, alpha)
		# file = open('store/sim_is.txt', 'rb')
		# pickle.dump(sim_is, file)
		# # sim_is = pickle.load(file)
		# file.close()
		# file = open('store/weighted_sim.txt', 'rb')
		# pickle.dump(weighted_sim, file)
		# # weighted_sim = pickle.load(file)
		# file.close()

		print("\nComputing User Similarities")
		similarity_list = similarity(weighted_sim, users, seq, common_items, item_ratings)
		# file = open('store/similarity_list.txt', 'rb')
		# # similarity_list = pickle.load(file)
		# pickle.dump(similarity_list, file)
		# file.close()
		
		print("\nPredicting Ratings")
		predictions, rmse, mae = rating_prediction(df_test, ratings, item_ratings, similarity_list, K)
		# file = open('store/predictions.txt', 'rb')
		# pickle.dump(predictions, file)
		# # test_set = pickle.load(file)
		# file.close()		
		# file = open('store/rmse.txt', 'rb')
		# pickle.dump(rmse, file)
		# # rmse = pickle.load(file)
		# file.close()
		# file = open('store/mae.txt', 'rb')
		# pickle.dump(mae, file)
		# # mae = pickle.load(file)
		# file.close()
		

		[precision, recall, F_score] = precision_recall_calculation(predictions, threshold=3.5)
		print("\n" + "-"*50)
		print("alpha = ", alpha, "; K = ", K)
		print("RMSE:", rmse)
		print("MAE:", mae)
		print("Precision: ", precision)
		print("Recall: ", recall)
		print("F-Score: ",F_score)
		print("-"*50)
		# print(str(rmse) + "\t" + str(mae) + "\t" + str(precision) + "\t" + str(recall) + "\t" + str(F_score))


				


if __name__ == "__main__":
	# We do not want the data to be sampled everytime, else the predictions won't match with each other.
	mode = "Test"
	df_train, df_test = data.get_train_test_data(new_sample = False)
	# Hyperparameters
	alpha = 0.8
	K = 1
	interest_sequence(df_train, df_test, mode, alpha, K)
