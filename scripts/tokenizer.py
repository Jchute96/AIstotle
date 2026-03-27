import json

# TODO handle unknown tokens that aren't in vocab

class BPETokenizer:
    
    def __init__(self):
        
        # Vocab dictionary used to store token strings and their corresponding int ids
        self.vocab = {}
        # List that holds the merge rules for each token
        self.merge_list = []
    
    
    # Build the vocabulary and merge rules
    def train(self, text, vocab_size):
        
        # Get every unique char in the text to add to our initial vocabulary and sort it
        unique_chars = sorted(set(text))
        
        # Add each unique char to our vocab dictionary along with its id number starting from zero
        for int_id, char in enumerate(unique_chars):
            self.vocab[char] = int_id
        
        # Convert the text into a list of chars that will be our tokens
        tokens = list(text)
        
        # Keep adding new tokens and tuples to our vocab and merge_list until we reach the preset vocab size
        while len(self.vocab) < vocab_size:
            
            # Dictionary to store the count for each pair of tokens seen
            pair_counts = {}
        
            # Iterate through the tokens and count how many times each token pair is seen
            for i in range(len(tokens) - 1):
            
                # If we have seen this pair of tokens before add 1 to its count
                if (tokens[i], tokens[i+1]) in pair_counts:
                    pair_counts[(tokens[i], tokens[i+1])] += 1
                
                # If we have not seen this pair before save the tokens as a tuple in the dict and set its value to 1
                else:
                    pair_counts[(tokens[i], tokens[i+1])] = 1
        
            # Get the value for every key in the dictionary and return the key with the biggest value  
            most_seen_pair = max(pair_counts, key=pair_counts.get)
            
            # If the best pair only appears once that means there are no more good merges so break
            if pair_counts[most_seen_pair] == 1:
                break
        
            # Create a new token by merging the most seen pair together
            new_token = most_seen_pair[0] + most_seen_pair[1]
        
            # Add the new token to the vocabulary and create an id using the next available int
            self.vocab[new_token] = len(self.vocab)
        
            # Append the most seen pair tuple to our merge list
            self.merge_list.append(most_seen_pair)
        
            # List to store our new merged tokens
            merged_tokens = []
            index = 0
        
            # Iterate through the tokens and replace our most_seen_pair of tokens with their merged version
            while index < len(tokens) - 1:
            
                # If current tokens match our most seen pair merge them, add them to merged list, and skip the next index
                if (tokens[index], tokens[index+1]) == most_seen_pair:
                    merged_tokens.append(tokens[index] + tokens[index+1])
                    index += 2
                
                # Else just add it like normal
                else:
                    merged_tokens.append(tokens[index])
                    index += 1 
            
            # Add last token if it was not appended
            if index == len(tokens) - 1:
                merged_tokens.append(tokens[index])
        
            # Set tokens to the new merged tokens
            tokens = merged_tokens     
    
    
    # Convert a users text into integer ids
    def encode(self, text):
        
        token_ids = []
        
        # Break the text into a list of individual chars
        tokens = list(text)
        
        # Iterate through the merges we have seen in our merge list
        for merge in self.merge_list:
            
            index = 0
            merged_tokens = []
            
            # Iterate through the users text and if we recognize a char pattern we have seen before in our merge list perform the merge on it 
            # and add to merged tokens list, otherwise just leave the char alone and add it
            while index < len(tokens) - 1:
                
                if (tokens[index], tokens[index+1]) == merge:
                    merged_tokens.append(tokens[index] + tokens[index+1])
                    index += 2
                
                else:
                    merged_tokens.append(tokens[index])
                    index += 1
            
            # Add last token if it was not appended
            if index == len(tokens) - 1:
                merged_tokens.append(tokens[index])
            
            # Set tokens to the new merged tokens
            tokens = merged_tokens
 
        # Get the corresponding int ids for each token in our users text
        for token in tokens:
            token_ids.append(self.vocab[token])
            
        # Return the encoded values of the users text
        return token_ids
    
          
    # Convert the int ids back into a string
    def decode(self, ids):
        
        decoded = ''
        
        # Create a new dictionary with the id and token strings swapped so we can easily look up id numbers
        reverse_vocab = {value: key for key, value in self.vocab.items()}
        
        # Add the decoded strings to our return value
        for id in ids:
            decoded += reverse_vocab[id]
        
        return decoded
    
    
    # Save the trained vocab and merge rules to a json file
    def save(self, filepath):
        
        # Create the dictionary with vocab and merge_list to save
        data = {"vocab": self.vocab, "merge_list": self.merge_list}
        
        # Save the new dictionary to a file and use json.dump to make it a json formatted string
        with open(filepath, "w") as file:
            json.dump(data, file)
     
            
    # Load a previous training data from a json file
    def load(self, filepath):
        
        # Read the JSON file from filepath
        with open(filepath, "r") as file:
            data = json.load(file)
            
        # Restore self.vocab and self.merge_list and revert merge list back to tuples
        self.vocab = data["vocab"]
        self.merge_list = [tuple(merge) for merge in data["merge_list"]]