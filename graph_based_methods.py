# The following code is adapted from https://github.com/jiangjmj/Graph-based-Uncertainty with minor modifications


def _get_verbalized_confidence(self, gen_id, claim_id, data, new_breakdown):
        vc, vc_raw_result = utils.get_verbalized_confidence(
            question=data['entity'],
            claim=new_breakdown,
            model_instance=self.llm_model,
            problem_type='fact',
            with_options=True,
        )
        return vc, vc_raw_result

VC_OPTIONS = {'nochance': 0, 'littlechance': 0.2, 'lessthaneven': 0.4, 'fairlypossible': 0.6, 'verygoodchance': 0.8, 'almostcertain': 1.0}

def parse_confidence(input_string, with_options=False):
    """
    Parses the input string to find a percentage or a float between 0.0 and 1.0 within the text.
    If a percentage is found, it is converted to a float.
    If a float between 0.0 and 1.0 is found, it is also converted to a float.
    In other cases, returns -1.

    :param input_string: str, the string to be parsed.
    :return: float, the parsed number or -1 if no valid number is found.
    """
    if with_options:
        split_list = input_string.split(':')
        if len(split_list) > 1:
            input_string = split_list[1]
        only_alpha = re.sub(r'[^a-zA-Z]', '', input_string).lower()
        if only_alpha in VC_OPTIONS:
            return VC_OPTIONS[only_alpha]
    else:
        # Search for a percentage in the text
        percentage_match = re.search(r'(\d+(\.\d+)?)%', input_string)
        if percentage_match:
            return float(percentage_match.group(1)) / 100

        # Search for a float between 0.0 and 1.0 in the text
        float_match = re.search(r'\b0(\.\d+)?\b|\b1(\.0+)?\b', input_string)
        if float_match:
            return float(float_match.group(0))

    # If neither is found, return -1
    return -1

def get_verbalized_confidence(question, claim, model_instance, problem_type='qa', with_options=False):
    if problem_type == 'qa':
        prompt = f'You are provided with a question and a possible answer. Provide the probability that the possible answer is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\nProbability: <the probability that your guess is correct as a percentage, without any extra commentary whatsoever; just the probability!>\n\nThe question is: {question}\nThe possible answer is: {claim}'
    elif problem_type == 'fact':
        prompt = f'You are provided with some possible information about a person. Provide the probability that the information is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\nProbability: <the probability that your guess is correct as a percentage, without any extra commentary whatsoever; just the probability!>\n\nThe person is: {question}\nThe possible information is: {claim}'
    
    if with_options:
        if problem_type == 'qa':
            prompt = f'You are provided with a question and a possible answer. Describe how likely it is that the possible answer is correct as one of the following expressions:\nNo chance (0%)\nLittle chance  (20%)\nLess than even (40%)\nFairly possible (60%)\nVery good chance (80%)\nAlmost certain (100%)\n\nGive ONLY your confidence phrase, no other words or explanation. For example:\n\nConfidence: <description of confidence, without any extra commentary whatsoever; just a short phrase!>\n\nThe question is: {question}\nThe possible answer is: {claim}'
        elif problem_type == 'fact':
            prompt = f'You are provided with some possible information about a person. Describe how likely it is that the possible answer is correct as one of the following expressions:\nNo chance (0%)\nLittle chance  (20%)\nLess than even (40%)\nFairly possible (60%)\nVery good chance (80%)\nAlmost certain (100%)\n\nGive ONLY your confidence phrase, no other words or explanation. For example:\n\nConfidence: <description of confidence, without any extra commentary whatsoever; just a short phrase!>\n\nThe person is: {question}\nThe possible information is: {claim}'
            
        results = model_instance.generate_given_prompt(prompt)

    if 'generation' in results:
        generation = results['generation']
    else:
        choice = results["choices"][0]
        generation = choice['message']['content'].strip()
    confidence = parse_confidence(generation, with_options=with_options)

    return confidence, results


def evaluate_single_match(self, instance_data, generation_id):
    claims = instance_data['breakdown'].copy()
    all_match_lists = []
    raw_match_results = []

    # Combine the most likely generation with a limited number of generated texts
    all_generations = [instance_data['most_likely_generation']] + instance_data['more_generations'][:self.args.sc_samples]

    for gen_index, generation in enumerate(all_generations):
        current_match_list = []
        current_raw_match_results = []

        for claim_index, claim in enumerate(claims):
            # Check if the result is already cached
            if (generation_id < len(self.cached_results['match']) and
                gen_index < len(self.cached_results['match'][generation_id]) and
                claim_index < len(self.cached_results['match'][generation_id][gen_index])):
                match_raw_result = self.cached_results['match'][generation_id][gen_index][claim_index]
            else:
                match_raw_result = self.perform_faithfulness_check(
                    generation=generation,
                    claim=claim,
                )

            # Parse the response
            if 'choices' in match_raw_result['return']:
                choice = match_raw_result['return']["choices"][0]
                response = choice['message']['content'].strip()
            else:
                response = match_raw_result['return'].strip()

            # Clean and validate the response
            response = ''.join([char for char in response if char.isalpha()]).lower()
            response = 'no' if 'notsupported' in response else response
            if response not in ['yes', 'no']:
                print(f'Invalid response: {response}, gen_index: {gen_index}, claim_index: {claim_index}')
            current_match_list.append(float(response == 'yes'))
            current_raw_match_results.append(match_raw_result)

        all_match_lists.append(current_match_list)
        raw_match_results.append(current_raw_match_results)

    # Update the cached results
    if generation_id >= len(self.cached_results['match']):
        self.cached_results['match'].append(raw_match_results)
    elif len(raw_match_results) > len(self.cached_results['match'][generation_id]):
        self.cached_results['match'][generation_id] = raw_match_results

    return all_match_lists


    def evaluate_all_matches(self):
        all_breakdown_matches = self.source_data.copy()

        for generation_id, instance_data in tqdm(enumerate(self.source_data), "Constructing Edges"):
            # the list of "is_supported" for claim
            match_lists = self.evaluate_single_match(instance_data, generation_id)

            # Save the cached results
            with open(self.raw_results_path, 'w') as f:
                json.dump(self.cached_results, f, indent=2)

            match_lists = np.array(match_lists)
            # most likely generation + diverse generations
            assert np.shape(match_lists) == (self.args.sc_samples + 1, len(instance_data['breakdown']))
            average_scores = np.mean(match_lists, axis=0)

            all_breakdown_matches[generation_id][f'sc_match_{self.args.sc_samples}samples'] = match_lists.tolist()
            verbalized_confidence_list = np.zeros(len(instance_data['breakdown']))
            verbalized_confidence_with_options_list = np.zeros(len(instance_data['breakdown']))

            for claim_index, pointwise_dict in enumerate(instance_data['pointwise_dict']):
                verbalized_confidence_with_options_list[claim_index] = pointwise_dict['verbalized_confidence_with_options']
                verbalized_confidence_list[claim_index] = pointwise_dict['inline_verbalized_confidence']

            # Calculate centrality scores
            centrality_scores_vcwo = utils.calculate_bg_centrality(match_lists, len(instance_data['breakdown']), vc_lst=verbalized_confidence_with_options_list)
            centrality_scores_vc = utils.calculate_bg_centrality(match_lists, len(instance_data['breakdown']), vc_lst=verbalized_confidence_list)

            for claim_index, pointwise_dict in enumerate(instance_data['pointwise_dict']):
                assert pointwise_dict['claim'] == instance_data["breakdown"][claim_index]
                pointwise_dict[f'sc_score_{self.args.sc_samples}samples'] = average_scores[claim_index]

                for centrality_name, centrality_score in centrality_scores_vcwo.items():
                    pointwise_dict[f'breakdown_{centrality_name}_{self.args.sc_samples}samples'] = centrality_score[claim_index]

                pointwise_dict[f'sc_plus_vc'] = average_scores[claim_index] + pointwise_dict['verbalized_confidence_with_options']
                pointwise_dict[f'sc_based_vc'] = average_scores[claim_index] + pointwise_dict['verbalized_confidence_with_options'] * 0.1
                
                # A baseline uses neglectable extra computation than sc, using inline verbalized confidence to break ties in sc 
                pointwise_dict[f'sc_based_ilvc'] = average_scores[claim_index] + pointwise_dict['inline_verbalized_confidence'] * 0.1 
                
        # Save the final results
        with open(self.collected_results_path, 'w') as outfile:
            json.dump(all_breakdown_matches, outfile, indent=4)

        return all_breakdown_matches


    def calculate_bg_centrality(lsts, length, vc_lst=None):
        centrality_dict = {}
        for centrality_name in CENTRALITY_TYPE_LIST:
            centrality_dict[centrality_name] = np.ones(length) * -1    
        
        filtered_lists = np.array(lsts)
        gen_num, flitered_breakdown_len = np.shape(filtered_lists)[0], np.shape(filtered_lists)[1]
        adjacency_matrix = np.zeros((gen_num + flitered_breakdown_len, gen_num + flitered_breakdown_len))
        adjacency_matrix[:gen_num, gen_num:] = filtered_lists
        adjacency_matrix[gen_num:, :gen_num] = filtered_lists.T
        G = nx.from_numpy_array(adjacency_matrix)
        if vc_lst is not None:
            combined_c = bg_closeness_centrality_with_node_confidence(G, gen_num, vc_lst)

        centrality = np.ones(length) * -1 
        for function_name in centrality_dict:
            try:
                if function_name in ['eigenvector_centrality', 'pagerank']:
                    centrality = getattr(nx, function_name)(G, max_iter=5000)
                elif function_name == 'closeness_centrality_with_node_confidence' and vc_lst is not None:
                    centrality = combined_c
                else:
                    centrality = getattr(nx, function_name)(G)
            except:
                pass

            assert length == flitered_breakdown_len
            centrality = [centrality[i] for i in sorted(G.nodes())] # List of scores in the order of nodes
            centrality_dict[function_name] = centrality[gen_num:]
                    
        return centrality_dict

    def bg_closeness_centrality_with_node_confidence(G, gen_num, verb_conf):
        n = len(G)
        combined_centrality = np.ones(len(G)) * -1
        
        for i in range(gen_num, n):
            sum_shortest_path_to_gens, sum_vc, sum_shortest_path_to_gens_vc_unweighted, sum_shortest_path_to_gens_vc_product = 0, 0, 0, 0
            reachable = 0
            
            for gen_id in range(n):
                if i == gen_id:
                    continue
                try:
                    shortest_path = nx.shortest_path(G, source=i, target=gen_id)
                    shortest_path = [node for node in shortest_path if node >= gen_num]
                    verb_conf_sum = np.sum([1 - verb_conf[node - gen_num] for node in shortest_path])
                                    
                    path_length = nx.shortest_path_length(G, source=i, target=gen_id)
                    
                    sum_vc += verb_conf_sum
                    sum_shortest_path_to_gens += path_length + verb_conf_sum
                    
                    sum_shortest_path_to_gens_vc_unweighted += path_length + np.sum([1 - verb_conf[node - gen_num] for node in shortest_path[1:]])
                    reachable += 1
                except nx.NetworkXNoPath:
                    continue
            
            scaling_factor = reachable / (n - 1)
            combined_centrality[i] = scaling_factor * reachable / sum_shortest_path_to_gens if reachable > 0 else 0

        return combined_centrality