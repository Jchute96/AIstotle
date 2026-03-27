# Train - learn the merge rules from the dataset
# Encode - turn text into a list of numbers
# Decode - turn a list of numbers back into text


class BPETokenizer:
    
    
    def __init__(self):
        
        # Vocab dictionary used to store token strings and their corresponding int ids
        self.vocab = {}
        # List that holds the merge rules for each token
        self.merge_list = []
    
    
    def train(self, text, vocab_size):
        
        # Get every unique char in the text to add to our initial vocabulary and sort it
        unique_chars = sorted(set(text))
        
        # Add each unique char to our vocab dictionary along with its id number starting from zero
        for int_id, char in enumerate(unique_chars):
            self.vocab[char] = int_id
        
        # Convert the text into a list of chars that will be our tokens
        tokens = list(text)
        
        # Keep adding new tokens and tupples to our vocab and merge_list until we reach the preset vocab size
        while len(self.vocab) < vocab_size:
            
            # Dictionary to store the count for each pair of tokens seen
            pair_counts = {}
        
            # Iterate through the tokens and count how many times each token pair is seen
            for i in range(len(tokens) - 1):
            
                # If we have seen this pair of tokens before add 1 to its count
                if (tokens[i], tokens[i+1]) in pair_counts:
                    pair_counts[(tokens[i], tokens[i+1])] += 1
                
                # If we have not seen this pair before save the tokens as a tupple in the dict and set its value to 1
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
        
            # Append the most seen pair tupple to our merge list
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
    
    
    # Convert a users text into integers that are mapped in our vocabulary
    def encode(self, text):
        
        token_ids = []
        
        # Break the text into a list of individual chars
        tokens = list(text)
        
        # Iterate through the merges we have seen in our merge list
        for merge in self.merge_list:
            
            index = 0
            merged_tokens = []
            last_char_merged = False
            
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
            
    
    def decode(self, ids):
        # Convert list of integer IDs back to text
        pass
    


tokenizer = BPETokenizer()

tokenizer.train("low lower newest widest low lower low lowest newest", vocab_size=20)
print(tokenizer.vocab)
print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------')
print(tokenizer.merge_list)

encoded = tokenizer.encode("lowest")
print(encoded)
    
    