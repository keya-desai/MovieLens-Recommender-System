import pandas as pd
import numpy as np
from collections import defaultdict
from surprise import Reader, Dataset
from surprise import SVD, accuracy, SVDpp, SlopeOne, BaselineOnly, CoClustering, KNNBasic
import data
import time
import pickle

def convert_traintest_dataframe_forsurprise(training_dataframe, testing_dataframe):
	training_dataframe = training_dataframe.iloc[:, :-1]
	testing_dataframe = testing_dataframe.iloc[:, :-1]
	reader = Reader(rating_scale=(0,5))
	trainset = Dataset.load_from_df(training_dataframe[['userId', 'movieId', 'rating']], reader)
	testset = Dataset.load_from_df(testing_dataframe[['userId', 'movieId', 'rating']], reader)
	trainset = trainset.construct_trainset(trainset.raw_ratings)
	testset=testset.construct_testset(testset.raw_ratings)
	return([trainset,testset])

def compute_error(actual_ratings, estimate_ratings):
	ratings = np.array(actual_ratings)
	estimate = np.array(estimate_ratings)

	rmse = np.sqrt(np.sum(np.square(np.subtract(ratings, estimate)))/np.size(ratings))
	mae = np.sum(np.abs(np.subtract(ratings, estimate)))/np.size(ratings)

	return rmse, mae

def svdalgorithm(trainset, testset):

	print("\n" + "-" *5 + " SVD algorithm using surprise package " + "-" *5)
	algo = SVD()
	algo.fit(trainset)
	predictions = algo.test(testset)
	rmse = accuracy.rmse(predictions)
	mae = accuracy.mae(predictions)
	return rmse, mae, predictions


def baseline(trainset, testset):

	print("\n" + "-" *5 + " Baseline algorithm using surprise package " + "-" *5)
	algo = BaselineOnly()
	algo.fit(trainset)
	predictions = algo.test(testset)
	rmse = accuracy.rmse(predictions)
	mae = accuracy.mae(predictions)
	return rmse, mae, predictions

def svdpp(trainset, testset):
	# Matrix factorization - SVD++
	print("\n" + "-" *5 + " SVD++ algorithm using surprise package " + "-" *5)
	algo = SVDpp()
	algo.fit(trainset)
	predictions = algo.test(testset)
	rmse = accuracy.rmse(predictions)
	mae = accuracy.mae(predictions)
	return rmse, mae, predictions

def slopeOne(trainset, testset):
	# Slope One
	print("\n" + "-" *5 + " SlopeOne algorithm using surprise package " + "-" *5)
	algo = SlopeOne()
	algo.fit(trainset)
	predictions = algo.test(testset)
	rmse = accuracy.rmse(predictions)
	mae = accuracy.mae(predictions)
	return rmse, mae, predictions

def coClustering(trainset, testset):
	# CoClustering
	print("\n" + "-" *5 + " CoClustering algorithm using surprise package " + "-" *5)
	algo = CoClustering()
	algo.fit(trainset)
	predictions = algo.test(testset)
	rmse = accuracy.rmse(predictions)
	mae = accuracy.mae(predictions)
	return rmse, mae, predictions

def kNNBasic(trainset, testset):
	# KNN basic
	print("\n" + "-" *5 + " KNNBasic algorithm using surprise package " + "-" *5)
	sim_options = {
	                'name': 'MSD',      # MSD similarity measure gives the best result
	              #  'user_based': True  # compute  similarities between users: MAE = 0.7744112391896695
	               'user_based': False  # compute  similarities between items: MAE = 0.7685376263051
	               }
	algo = KNNBasic(sim_options = sim_options)
	# algo = KNNBasic()
	algo.fit(trainset)
	predictions = algo.test(testset)
	rmse = accuracy.rmse(predictions)
	mae = accuracy.mae(predictions)
	return rmse, mae, predictions

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

#Writes the predictions of all approaches to text files
def hybrid_approach(trainset, testset, phase = "Test", weights = []):
	if phase == "Train":
		start_time = time.time()
		rmse_arr = []
		mae_arr = []

		rmse, mae, prediction_baseline = baseline(trainset, testset)
		file = open('store/prediction_baseline.txt', 'wb')
		pickle.dump(prediction_baseline, file)
		file.close()
		rmse_arr.append(rmse)
		mae_arr.append(mae)
		print("Elapsed Time: ", time.time() - start_time)

		rmse, mae, predictions_svd = svdalgorithm(trainset, testset)
		file = open('store/predictions_svd.txt', 'wb')
		pickle.dump(predictions_svd, file)
		file.close()
		rmse_arr.append(rmse)
		mae_arr.append(mae)
		print("Elapsed Time: ", time.time() - start_time)

		rmse, mae, predictions_svdpp = svdpp(trainset, testset)
		file = open('store/predictions_svdpp.txt', 'wb')
		pickle.dump(predictions_svdpp, file)
		file.close()
		rmse_arr.append(rmse)
		mae_arr.append(mae)
		print("Elapsed Time: ", time.time() - start_time)
		
		rmse, mae, predictions_s1 = slopeOne(trainset, testset)
		file = open('store/predictions_s1.txt', 'wb')
		pickle.dump(predictions_s1, file)
		file.close()
		rmse_arr.append(rmse)
		mae_arr.append(mae)
		print("Elapsed Time: ", time.time() - start_time)
		
		rmse, mae, predictions_co = coClustering(trainset, testset)
		file = open('store/predictions_co.txt', 'wb')
		pickle.dump(predictions_co, file)
		file.close()
		rmse_arr.append(rmse)
		mae_arr.append(mae)
		print("Elapsed Time: ", time.time() - start_time)
		
		rmse, mae, predictions_knn = kNNBasic(trainset, testset)
		file = open('store/predictions_knn.txt', 'wb')
		pickle.dump(predictions_knn, file)
		file.close()
		rmse_arr.append(rmse)
		mae_arr.append(mae)
		print("Elapsed Time: ", time.time() - start_time)

		file = open('store/mae_arr.txt', 'wb')
		pickle.dump(mae_arr, file)
		file.close()

		file = open('store/rmse_arr.txt', 'wb')
		pickle.dump(rmse_arr, file)
		file.close()

	if phase == "Test":
		predictions_all = []

		file = open('store/prediction_baseline.txt', 'rb')
		temp = pickle.load(file)
		predictions_all.append(temp)
		file.close()

		file = open('store/predictions_svd.txt', 'rb')
		predictions_all.append(pickle.load(file))
		file.close()

		file = open('store/predictions_svdpp.txt', 'rb')
		predictions_all.append(pickle.load(file))
		file.close()

		file = open('store/predictions_s1.txt', 'rb')
		predictions_all.append(pickle.load(file))
		file.close()

		file = open('store/predictions_co.txt', 'rb')
		predictions_all.append(pickle.load(file))
		file.close()

		file = open('store/predictions_knn.txt', 'rb')
		predictions_all.append(pickle.load(file))
		file.close()

		file = open('store/mae_arr.txt', 'rb')
		mae_arr = pickle.load(file)
		file.close()

		file = open('store/rmse_arr.txt', 'rb')
		rmse_arr = pickle.load(file)
		file.close()

		actual_ratings = []
		estimate_arr = []

		for p in predictions_all[1]:
			actual_ratings.append(p[2])

		for i, predictions in enumerate(predictions_all):
			estimate_arr.append([])
			for p in predictions:
				estimate_arr[i].append(p[3])

		if len(weights) == 0:
			total = 0
			for i, (e,f) in enumerate(zip(rmse_arr, mae_arr)):
				if i in [0, 1, 2, 3, 4, 5]:
					total += (1)/((e) ** 1)

			for i, (e,f) in enumerate(zip(rmse_arr, mae_arr)):
				if i in [0, 1, 2, 3, 4, 5]:
					weights.append((1)/(((e) ** 1) * total))
				else:
					weights.append(0)

			hybrid_estimates = np.zeros(np.asarray(estimate_arr[0]).shape)

			for i, estimate in enumerate(estimate_arr):
				hybrid_estimates += np.multiply(estimate, weights[i])

		print(weights)

		hybrid_predictions = []

		for p, h in zip(predictions_all[0], hybrid_estimates):
			hybrid_predictions.append((p[0], p[1], p[2], h, p[4]))

		rmse, mae = compute_error(actual_ratings, hybrid_estimates)
		[precision, recall, F_score] = precision_recall_calculation(hybrid_predictions, threshold=3.5)

		print("\n" + "-" *5 + " Hybrid algorithm " + "-" *5)
		print("RMSE: ", rmse)
		print("MAE: ", mae)
		print("Precision: ", precision)
		print("Recall: ", recall)
		print("F-Score: ",F_score)

		print(str(rmse) + "\t" + str(mae) + "\t" + str(precision) + "\t" + str(recall) + "\t" + str(F_score))
	
if __name__ == "__main__":
	#We do not want the data to be sampled everytime, else the predictions won't match with each other.
	trainset = []
	testset = []
	phase = "Test"
	weights = []
	if phase == "Train":
		df_train, df_test = data.get_train_test_data(new_sample = False)
		trainset, testset = convert_traintest_dataframe_forsurprise(df_train, df_test)
	hybrid_approach(trainset, testset, phase, weights)