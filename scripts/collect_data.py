import requests
import re

# Urls to the philosophy texts
urls = {
    "meditations": "https://www.gutenberg.org/cache/epub/2680/pg2680.txt",
    "enchiridion": "https://www.gutenberg.org/cache/epub/45109/pg45109.txt",
    "seneca_morals": "https://www.gutenberg.org/cache/epub/56075/pg56075.txt",
    "zarathustra": "https://www.gutenberg.org/cache/epub/1998/pg1998.txt",
    "nicomachean_ethics": "https://www.gutenberg.org/cache/epub/8438/pg8438.txt",
    "the_republic": "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",
    "beyond_good_and_evil": "https://www.gutenberg.org/cache/epub/4363/pg4363.txt",
    "the_prince": "https://www.gutenberg.org/cache/epub/1232/pg1232.txt",
    "phaedo": "https://www.gutenberg.org/cache/epub/1658/pg1658.txt",
    "symposium": "https://www.gutenberg.org/cache/epub/1600/pg1600.txt",
    "art_of_war": "https://www.gutenberg.org/cache/epub/17405/pg17405.txt",
    "tusculan": "https://www.gutenberg.org/cache/epub/14988/pg14988.txt",
    "bacon_essays": "https://www.gutenberg.org/cache/epub/575/pg575.txt",
}

# Where each book's philosophy text actually start and end
boundaries = {
    "meditations":  ("Of my grandfather Verus", "Go thy ways then well pleased"),
    "enchiridion":  ("There are things which are within our power", "Thus Socrates became perfect"),
    "seneca_morals": ("It is, perhaps, one of the most pernicious", "not to oppress them"),
    "zarathustra":  ("When Zarathustra was thirty years old", "like a morning sun coming out of gloomy mountains"),
    "nicomachean_ethics": ("Every art, and every science reduced", "regulations are best for each"),
    "the_republic": ("I went down yesterday to the Piraeus with Glaucon", "in this life and in the pilgrimage of a thousand"),
    "beyond_good_and_evil": ("The Will to Truth, which is to tempt us to many a hazardous", "marvels of my solitude, you, my old, beloved--EVIL thoughts"),
    "the_prince": ("All states, all powers, that have held", "should be born, not in Lucca"),
    "phaedo": ("The doctrine of the immortality of the soul has sunk deep", "he was the"),
    "symposium": ("Concerning the things about which you ask to be informed","In the evening he retired to rest"),
    "art_of_war": ("The art of war is of vital importance to the State.","Spies are a most important"),
    "tusculan": ("At a time when I had entirely, or to a great degree,","suddenly revives out of the most desperate"),
    "bacon_essays": ("WHAT is truth? said jesting Pilate","have as great a watch"),
}

# Get the raw_text text within the range we need
def extract_content(text, start_text, end_text):
    
    # Get the indexes of the first and last text that we actually want
    start_index = text.find(start_text)
    end_index = text.find(end_text)
    
    # Get all the text from the beginning of the start index to the end of the end index
    return text[start_index:end_index + len(end_text)]


def clean_text(text):
    
    # Replace all occurences of the windows new line representation to make everything consistent
    text = text.replace('\r\n', '\n')
    
    # Remove anything that is a set of brackets and any characters within
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove underscores
    text = re.sub(r'_+', '', text)
    
    # Remove lines that start with a roman numeral and consist of spaces then capital letters and then a period
    text = re.sub(r'^[IVXLCDM]+\.\s+[A-Z\s]+\.$', '', text, flags=re.MULTILINE)
    
    # Remove lines that start with CHAPTER and are followed by a roman numeral
    text = re.sub(r'^CHAPTER\s+[IVXLCDM]+\.\s*$', '', text, flags=re.MULTILINE)
    
    # Remove lines that start with chapter and have a title after the roman numerals
    text = re.sub(r'^CHAPTER\s+[IVXLCDM]+\.\s+[A-Z\s]+$', '', text, flags=re.MULTILINE)
    
    # Remove lines that start with not all caps Chapter and are followed by a roman numeral
    text = re.sub(r'^Chapter\s+[IVXLCDM]+\.\s*$', '', text, flags=re.MULTILINE)
    
    
    # Remove lines that are at least 10 characters long and consist of only capital letters and spaces
    text = re.sub(r'^[A-Z\s]{10,}$', '', text, flags=re.MULTILINE)
    
    # Remove roman numeral lines that are by themselves
    text = re.sub(r'^[IVXLCDM]+\.\s*$', '', text, flags=re.MULTILINE)

    # Remove digits that are by themselves
    text = re.sub(r'^\d+\.\s*$', '', text, flags=re.MULTILINE)
    
    # Remove roman numeral prefixes from beginning of lines
    text = re.sub(r'^[IVXLCDM]+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove number prefixes from beginning of lines
    text = re.sub(r'^[0-9]+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove lines that are just asterisks and spaces
    text = re.sub(r'^\s*\*[\s\*]*$', '', text, flags=re.MULTILINE)

    # Condense the blank lines we created
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

# List to hold all of the clean text
all_clean_text = []

# Iterate through each url and add their text to a file
for name, url in urls.items():
    
    # Go to the url and get all of the text from it
    response = requests.get(url)
    raw_text = response.text
    
    # Extract only the actual book content within our boundaries
    start = boundaries[name][0]
    end = boundaries[name][1]
    content = extract_content(raw_text, start, end)
    
    # Clean the content
    cleaned_text = clean_text(content)
    
    # Create a file for writing
    with open(f"data/raw/{name}.txt", "w") as file:

        # Take the cleaned_text and write it to the file
        file.write(cleaned_text)
    
    # Add the text to all_clean_text list
    all_clean_text.append(cleaned_text)
    print(f"{name} has been saved to the list!\n")

# Combine all data into one file
dataset = "\n\n".join(all_clean_text)

with open("data/dataset.txt", "w") as file:
    file.write(dataset)

print(f"Dataset saved!\n")
    