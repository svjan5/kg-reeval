"""
	def get_validation_pred(self, args, model, unique_entities, eval_type='last'):
		average_hits_at_100_head, average_hits_at_100_tail = [], []
		average_hits_at_ten_head, average_hits_at_ten_tail = [], []
		average_hits_at_three_head, average_hits_at_three_tail = [], []
		average_hits_at_one_head, average_hits_at_one_tail = [], []
		average_mean_rank_head, average_mean_rank_tail = [], []
		average_mean_recip_rank_head, average_mean_recip_rank_tail = [], []

		# eval_type = 'rotate'
		"""
		From Shikhar
		I have implemented the following ways of evaulating the code:
			org:  	Original code used for evaluation by the authors
			last: 	Putting the correct entity at the end
			random: Putting it at a random position
			rotate:	Is what has been followed in https://github.com/KnowledgeBaseCompleter/eval-ConvKB 
			geq:	Greater than equal (just another variant)

		"""

		for iters in range(1):
			start_time = time.time()

			indices = [i for i in range(len(self.test_indices))]
			batch_indices = self.test_indices[indices, :]
			print("Sampled indices")
			print("test set length ", len(self.test_indices))
			entity_list = [j for i, j in self.entity2id.items()]

			ranks_head, ranks_tail = [], []
			reciprocal_ranks_head, reciprocal_ranks_tail = [], []
			hits_at_100_head, hits_at_100_tail = 0, 0
			hits_at_ten_head, hits_at_ten_tail = 0, 0
			hits_at_three_head, hits_at_three_tail = 0, 0
			hits_at_one_head, hits_at_one_tail = 0, 0

			for i in range(batch_indices.shape[0]):
				print(len(ranks_head))
				start_time_it = time.time()
				new_x_batch_head = np.tile(batch_indices[i, :], (len(self.entity2id), 1))
				new_x_batch_tail = np.tile(batch_indices[i, :], (len(self.entity2id), 1))

				# if(batch_indices[i, 0] not in unique_entities or batch_indices[i, 2] not in unique_entities):
				# 	continue

				new_x_batch_head[:, 0] = entity_list
				new_x_batch_tail[:, 2] = entity_list

				if eval_type != 'rotate':
					last_index_head = []  # array of already existing triples
					last_index_tail = []
					for tmp_index in range(len(new_x_batch_head)):
						temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1], new_x_batch_head[tmp_index][2])
						if temp_triple_head in self.valid_triples_dict.keys():
							last_index_head.append(tmp_index)

						temp_triple_tail = (new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][1], new_x_batch_tail[tmp_index][2])
						if temp_triple_tail in self.valid_triples_dict.keys():
							last_index_tail.append(tmp_index)

					# Deleting already existing triples, leftover triples are invalid, according
					# to train, validation and test data
					# Note, all of them maynot be actually invalid

					new_x_batch_head = np.delete(new_x_batch_head, last_index_head, axis=0)
					new_x_batch_tail = np.delete(new_x_batch_tail, last_index_tail, axis=0)

				# adding the current valid triples to the top, i.e, index 0

				if eval_type in ['org', 'geq']:
					new_x_batch_head = np.insert(new_x_batch_head, 0, batch_indices[i], axis=0)
					new_x_batch_tail = np.insert(new_x_batch_tail, 0, batch_indices[i], axis=0)

				elif eval_type in ['last']:
					new_x_batch_head = np.concatenate([new_x_batch_head, [batch_indices[i]]], 0)
					new_x_batch_tail = np.concatenate([new_x_batch_tail, [batch_indices[i]]], 0)

				elif eval_type in ['random']:
					rand_head 	 = np.random.randint(new_x_batch_head.shape[0])
					rand_tail 	 = np.random.randint(new_x_batch_tail.shape[0])

					new_x_batch_head = np.insert(new_x_batch_head, rand_head, batch_indices[i], axis=0)
					new_x_batch_tail = np.insert(new_x_batch_tail, rand_tail, batch_indices[i], axis=0)

				elif eval_type == 'rotate':
					entity_array_head = new_x_batch_head[:, 0]
					entity_array_tail = new_x_batch_tail[:, 2]
				else:
					raise NotImplementedError

				import math
				# Have to do this, because it doesn't fit in memory

				if 'WN' in args.data:
					num_triples_each_shot = int(
						math.ceil(new_x_batch_head.shape[0] / 4))

					scores1_head = model.batch_test(torch.LongTensor(new_x_batch_head[:num_triples_each_shot, :]).cuda())
					scores2_head = model.batch_test(torch.LongTensor(new_x_batch_head[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
					scores3_head = model.batch_test(torch.LongTensor(new_x_batch_head[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
					scores4_head = model.batch_test(torch.LongTensor(new_x_batch_head[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())
					# scores5_head = model.batch_test(torch.LongTensor(
					#     new_x_batch_head[4 * num_triples_each_shot: 5 * num_triples_each_shot, :]).cuda())
					# scores6_head = model.batch_test(torch.LongTensor(
					#     new_x_batch_head[5 * num_triples_each_shot: 6 * num_triples_each_shot, :]).cuda())
					# scores7_head = model.batch_test(torch.LongTensor(
					#     new_x_batch_head[6 * num_triples_each_shot: 7 * num_triples_each_shot, :]).cuda())
					# scores8_head = model.batch_test(torch.LongTensor(
					#     new_x_batch_head[7 * num_triples_each_shot: 8 * num_triples_each_shot, :]).cuda())
					# scores9_head = model.batch_test(torch.LongTensor(
					#     new_x_batch_head[8 * num_triples_each_shot: 9 * num_triples_each_shot, :]).cuda())
					# scores10_head = model.batch_test(torch.LongTensor(
					#     new_x_batch_head[9 * num_triples_each_shot:, :]).cuda())

					scores_head = torch.cat([scores1_head, scores2_head, scores3_head, scores4_head], dim=0)
					#scores5_head, scores6_head, scores7_head, scores8_head,
					# cores9_head, scores10_head], dim=0)
				else:
					scores_head = model.batch_test(new_x_batch_head)

					
				scores_head = scores_head.cpu().numpy().reshape(-1)

				if eval_type == 'org':
					sorted_indices_head = np.argsort(-scores_head, kind='stable')
					ranks_head.append(np.where(sorted_indices_head == 0)[0][0] + 1)

				elif eval_type == 'geq':
					ranks_head.append(np.sum(scores_head >= scores_head[0]))

				elif eval_type == 'last':
					sorted_indices_head = np.argsort(-scores_head, kind='stable')
					ranks_head.append(np.where(sorted_indices_head == sorted_indices_head.shape[0]-1)[0][0] + 1)

				elif eval_type == 'random':
					sorted_indices_head = np.argsort(-scores_head, kind='stable')
					ranks_head.append(np.where(sorted_indices_head == rand_head)[0][0] + 1)

				elif eval_type == 'rotate':
					scores_head	= scores_head.cpu().numpy()
					results		= np.reshape(scores_head, [entity_array_head.shape[0], 1])
					results_with_id	= np.hstack((np.reshape(entity_array_head, [entity_array_head.shape[0], 1]), results))
					results_with_id	= results_with_id[np.argsort(-results_with_id[:, 1])]
					results_with_id	= results_with_id[:, 0].astype(int)
					
					_filter = 0
					for tmpHead in results_with_id:
						if tmpHead == batch_indices[i][0]:
							break
						tmpTriple = (tmpHead, batch_indices[i][1], batch_indices[i][2])
						if tmpTriple in self.valid_triples_dict.keys():
							continue
						else:
							_filter += 1
					# else:
					# 	for tmpTail in results_with_id:
					# 		if tmpTail == batch_indices[i][2]:
					# 			break
					# 		tmpTriple = (batch_indices[i][0], batch_indices[i][1], tmpTail)
					# 		if (tmpTriple in train) or (tmpTriple in valid) or (tmpTriple in test):
					# 			continue
					# 		else:
					# 			_filter += 1

					ranks_head.append(_filter + 1)

				else:
					raise NotImplementedError

				reciprocal_ranks_head.append(1.0 / ranks_head[-1])

				# Tail part here

				if 'WN' in args.data:
					num_triples_each_shot = int(math.ceil(new_x_batch_tail.shape[0] / 4))

					scores1_tail = model.batch_test(torch.LongTensor(new_x_batch_tail[:num_triples_each_shot, :]).cuda())
					scores2_tail = model.batch_test(torch.LongTensor(new_x_batch_tail[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
					scores3_tail = model.batch_test(torch.LongTensor(new_x_batch_tail[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
					scores4_tail = model.batch_test(torch.LongTensor(new_x_batch_tail[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())
					# scores5_tail = model.batch_test(torch.LongTensor(
					#     new_x_batch_tail[4 * num_triples_each_shot: 5 * num_triples_each_shot, :]).cuda())
					# scores6_tail = model.batch_test(torch.LongTensor(
					#     new_x_batch_tail[5 * num_triples_each_shot: 6 * num_triples_each_shot, :]).cuda())
					# scores7_tail = model.batch_test(torch.LongTensor(
					#     new_x_batch_tail[6 * num_triples_each_shot: 7 * num_triples_each_shot, :]).cuda())
					# scores8_tail = model.batch_test(torch.LongTensor(
					#     new_x_batch_tail[7 * num_triples_each_shot: 8 * num_triples_each_shot, :]).cuda())
					# scores9_tail = model.batch_test(torch.LongTensor(
					#     new_x_batch_tail[8 * num_triples_each_shot: 9 * num_triples_each_shot, :]).cuda())
					# scores10_tail = model.batch_test(torch.LongTensor(
					#     new_x_batch_tail[9 * num_triples_each_shot:, :]).cuda())

					scores_tail = torch.cat([scores1_tail, scores2_tail, scores3_tail, scores4_tail], dim=0)
					#     scores5_tail, scores6_tail, scores7_tail, scores8_tail,
					#     scores9_tail, scores10_tail], dim=0)

				else:
					scores_tail = model.batch_test(new_x_batch_tail)

				scores_tail = scores_tail.cpu().numpy().reshape(-1)

				if eval_type == 'org':
					sorted_indices_tail = np.argsort(-scores_tail, kind='stable')
					ranks_tail.append(np.where(sorted_indices_tail == 0)[0][0] + 1)

				elif eval_type == 'geq':
					scores_tail = scores_tail.cpu().numpy()
					ranks_tail.append(np.sum(scores_tail >= scores_tail[0]))

				elif eval_type == 'last':
					sorted_indices_tail = np.argsort(-scores_tail, kind='stable')
					ranks_tail.append(np.where(sorted_indices_tail == sorted_indices_tail.shape[0]-1)[0][0] + 1)

				elif eval_type == 'random':
					sorted_indices_tail = np.argsort(-scores_tail, kind='stable')
					ranks_tail.append(np.where(sorted_indices_tail == rand_tail)[0][0] + 1)

				elif eval_type == 'rotate':
					scores_tail	= scores_tail.cpu().numpy()
					results		= np.reshape(scores_tail, [entity_array_tail.shape[0], 1])
					results_with_id	= np.hstack((np.reshape(entity_array_tail, [entity_array_tail.shape[0], 1]), results))
					results_with_id	= results_with_id[np.argsort(-results_with_id[:, 1])]
					results_with_id	= results_with_id[:, 0].astype(int)
					
					_filter = 0
					for tmpTail in results_with_id:
						if tmpTail == batch_indices[i][2]:
							break
						tmpTriple = (batch_indices[i][0], batch_indices[i][1], tmpTail)
						if tmpTriple in self.valid_triples_dict.keys():
							continue
						else:
							_filter += 1

					ranks_tail.append(_filter + 1)

				else:
					raise NotImplementedError

				reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
				print("sample - ", ranks_head[-1], ranks_tail[-1])

			for i in range(len(ranks_head)):
				if ranks_head[i] <= 100: 	hits_at_100_head 	= hits_at_100_head + 1
				if ranks_head[i] <= 10: 	hits_at_ten_head 	= hits_at_ten_head + 1
				if ranks_head[i] <= 3: 		hits_at_three_head 	= hits_at_three_head + 1
				if ranks_head[i] == 1: 		hits_at_one_head 	= hits_at_one_head + 1

			for i in range(len(ranks_tail)):
				if ranks_tail[i] <= 100: 	hits_at_100_tail 	= hits_at_100_tail + 1
				if ranks_tail[i] <= 10: 	hits_at_ten_tail 	= hits_at_ten_tail + 1
				if ranks_tail[i] <= 3: 		hits_at_three_tail 	= hits_at_three_tail + 1
				if ranks_tail[i] == 1: 		hits_at_one_tail 	= hits_at_one_tail + 1

			assert len(ranks_head) == len(reciprocal_ranks_head)
			assert len(ranks_tail) == len(reciprocal_ranks_tail)
			print("here {}".format(len(ranks_head)))
			print("\nCurrent iteration time {}".format(time.time() - start_time))
			print("Stats for replacing head are -> ")
			print("Current iteration Hits@100 are {}".format(hits_at_100_head / float(len(ranks_head))))
			print("Current iteration Hits@10 are {}".format(hits_at_ten_head / len(ranks_head)))
			print("Current iteration Hits@3 are {}".format(hits_at_three_head / len(ranks_head)))
			print("Current iteration Hits@1 are {}".format(hits_at_one_head / len(ranks_head)))
			print("Current iteration Mean rank {}".format(sum(ranks_head) / len(ranks_head)))
			print("Current iteration Mean Reciprocal Rank {}".format(sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)))

			print("\nStats for replacing tail are -> ")
			print("Current iteration Hits@100 are {}".format(hits_at_100_tail / len(ranks_head)))
			print("Current iteration Hits@10 are {}".format(hits_at_ten_tail / len(ranks_head)))
			print("Current iteration Hits@3 are {}".format(hits_at_three_tail / len(ranks_head)))
			print("Current iteration Hits@1 are {}".format(hits_at_one_tail / len(ranks_head)))
			print("Current iteration Mean rank {}".format(sum(ranks_tail) / len(ranks_tail)))
			print("Current iteration Mean Reciprocal Rank {}".format(sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)))

			average_hits_at_100_head.append(hits_at_100_head / len(ranks_head))
			average_hits_at_ten_head.append(hits_at_ten_head / len(ranks_head))
			average_hits_at_three_head.append(hits_at_three_head / len(ranks_head))
			average_hits_at_one_head.append(hits_at_one_head / len(ranks_head))
			average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))
			average_mean_recip_rank_head.append(sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))

			average_hits_at_100_tail.append(hits_at_100_tail / len(ranks_head))
			average_hits_at_ten_tail.append(hits_at_ten_tail / len(ranks_head))
			average_hits_at_three_tail.append(hits_at_three_tail / len(ranks_head))
			average_hits_at_one_tail.append(hits_at_one_tail / len(ranks_head))
			average_mean_rank_tail.append(sum(ranks_tail) / len(ranks_tail))
			average_mean_recip_rank_tail.append(sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail))

		print("\nAveraged stats for replacing head are -> ")
		print("Hits@100 are {}".format(sum(average_hits_at_100_head) / len(average_hits_at_100_head)))
		print("Hits@10 are {}".format(sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)))
		print("Hits@3 are {}".format(sum(average_hits_at_three_head) / len(average_hits_at_three_head)))
		print("Hits@1 are {}".format(sum(average_hits_at_one_head) / len(average_hits_at_one_head)))
		print("Mean rank {}".format(sum(average_mean_rank_head) / len(average_mean_rank_head)))
		print("Mean Reciprocal Rank {}".format(sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head)))

		print("\nAveraged stats for replacing tail are -> ")
		print("Hits@100 are {}".format(sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)))
		print("Hits@10 are {}".format(sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)))
		print("Hits@3 are {}".format(sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)))
		print("Hits@1 are {}".format(sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)))
		print("Mean rank {}".format(sum(average_mean_rank_tail) / len(average_mean_rank_tail)))
		print("Mean Reciprocal Rank {}".format(sum(average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)))

		cumulative_hits_100			= (sum(average_hits_at_100_head) / len(average_hits_at_100_head) + sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)) / 2
		cumulative_hits_ten			= (sum(average_hits_at_ten_head) / len(average_hits_at_ten_head) + sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)) / 2
		cumulative_hits_three		= (sum(average_hits_at_three_head) / len(average_hits_at_three_head) + sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)) / 2
		cumulative_hits_one			= (sum(average_hits_at_one_head) / len(average_hits_at_one_head) + sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)) / 2
		cumulative_mean_rank		= (sum(average_mean_rank_head) / len(average_mean_rank_head) + sum(average_mean_rank_tail) / len(average_mean_rank_tail)) / 2
		cumulative_mean_recip_rank	= (sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head) + sum(average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)) / 2

		print("\nCumulative stats are -> ")
		print("Hits@100 are {}".format(cumulative_hits_100))
		print("Hits@10 are {}".format(cumulative_hits_ten))
		print("Hits@3 are {}".format(cumulative_hits_three))
		print("Hits@1 are {}".format(cumulative_hits_one))
		print("Mean rank {}".format(cumulative_mean_rank))
		print("Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank))
"""