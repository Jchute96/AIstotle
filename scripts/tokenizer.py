import json
import re

      
class Node:
    
    def __init__(self, val):
        
        self.val = val
        self.prev = None
        self.next = None

   
# DoublyLinkedList class used to help track most frequent pairs to speed up training
class DoublyLinkedList():
    
    def __init__(self):
        self.head = None
        self.tail = None
    
    
    # Add nodes to the doubly linked list
    def append(self, val):
        curr_node = Node(val)
    
        if not self.head:
            self.head = curr_node
            self.tail = curr_node
        else:
            self.tail.next = curr_node
            curr_node.prev = self.tail
            self.tail = curr_node

            
def build_linked_list(tokens):
    tokens_doubly_list = DoublyLinkedList()
    
    for token in tokens:
        tokens_doubly_list.append(token)
        
    return tokens_doubly_list


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
        
        # Make it so [UNK] tokens start at index 0 in the vocab
        self.vocab["[UNK]"] = 0
        
        # Add each unique char to our vocab dictionary along with its id number starting from 1
        for int_id, char in enumerate(unique_chars, start=1):
            self.vocab[char] = int_id
            
        # Add the other special tokens used later for fine tuning
        special_tokens = ["[QUESTION]", "[ANSWER]"]
        
        for token in special_tokens:
            self.vocab[token] = len(self.vocab)
        
        # Convert the text into a list of chars that will be our tokens
        tokens = list(text)
        
        # Create a doubly linked list using the tokens to iterate through
        tokens_doubly_list = build_linked_list(tokens)
        
        # Dictionary to store the counts for each pair of tokens seen as well as the first node associated with the pair
        pair_counts = {}
        pair_nodes = {}
        
        curr_node = tokens_doubly_list.head
            
        # Iterate through the tokens and count how many times each token pair is seen
        while curr_node:
                
            if curr_node.next:
                    
                # If we have seen this pair of tokens before add 1 to its count
                if (curr_node.val, curr_node.next.val) in pair_counts:
                    pair_counts[(curr_node.val, curr_node.next.val)] += 1
                    pair_nodes[(curr_node.val, curr_node.next.val)].append(curr_node)
                    
                # If we have not seen this pair before save the tokens as a tuple in the dict and set its value to 1
                else:
                    pair_counts[(curr_node.val, curr_node.next.val)] = 1
                    pair_nodes[(curr_node.val, curr_node.next.val)] = [curr_node]
                
            curr_node = curr_node.next
        
        # Keep adding new tokens and tuples to our vocab and merge_list until we reach the preset vocab size
        while len(self.vocab) < vocab_size:
                
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

            # Most seen pair is merged so we must alter surrounding token counts and values to compensate for it
            for node in pair_nodes[most_seen_pair]:
                
                # Decrement count of the most seen pair since it has been removed
                pair_counts[most_seen_pair] -= 1
                
                # Decrement the old pair that was formed, increment new pair that is formed, and add new pair to pair_nodes
                if node.prev:
                    # Only decrement if pair still exists
                    if node.prev and (node.prev.val, most_seen_pair[0]) in pair_counts:
                        pair_counts[(node.prev.val, most_seen_pair[0])] -= 1
                    
                    if (node.prev.val, new_token) in pair_counts:
                        pair_counts[(node.prev.val, new_token)] += 1
                        pair_nodes[(node.prev.val, new_token)].append(node.prev)
                        
                    else:
                        pair_counts[(node.prev.val, new_token)] = 1
                        pair_nodes[(node.prev.val, new_token)] = [node.prev]
                                
                # Change the value of the node associated with the most seen pair to the new merged value
                node.val = new_token
                
                # Decrement the old pair that was formed by the last node of the merged pair and next node
                if node.next and node.next.next:
                    # Only decrement if pair still exists
                    if (most_seen_pair[1], node.next.next.val) in pair_counts:
                        pair_counts[(most_seen_pair[1], node.next.next.val)] -= 1

                # Skip over the merged node
                if node.next:
                    node.next = node.next.next
                
                # Change the prev of the next next node to the new valued node, increment the newly created pair, and add new pair to pair_nodes
                if node.next:
                    node.next.prev = node
                    
                    if (new_token, node.next.val) in pair_counts:
                        pair_counts[(new_token, node.next.val)] += 1
                        pair_nodes[(new_token, node.next.val)].append(node)
                    else:
                        pair_counts[(new_token, node.next.val)] = 1
                        pair_nodes[(new_token, node.next.val)] = [node]
                        
    
    # Convert a users text into integer ids
    def encode(self, text):
        
        token_ids = []
        special_tokens = ["[UNK]", "[QUESTION]", "[ANSWER]"]
        
        pattern = "(" + "|".join(re.escape(t) for t in special_tokens) + ")"
        
        # Splits the string into chunks based off of our special tokens
        chunks = re.split(pattern, text)
        
        for chunk in chunks:
            
            # Append the special token if it is seen in this chunk
            if chunk in special_tokens:
                token_ids.append(self.vocab[chunk])
            
            # If this chunk has something in it perform encoding
            elif chunk:
                
                # Break the chunk into a list of individual chars
                tokens = list(chunk)
                
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
 
                # Get the corresponding int ids for each token in our users text. If it was not seen before replace it with the UNK token value
                for token in tokens:
                    token_ids.append(self.vocab.get(token, self.vocab["[UNK]"]))
            
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
        
        
if __name__ == "__main__":
    
    with open("data/dataset.txt", "r") as file:
        text = file.read()
    
    tokenizer = BPETokenizer()
    
    tokenizer.train(text, 8000)
    
    tokenizer.save("data/tokenizer.json")
    
    print("Tokenization complete!")