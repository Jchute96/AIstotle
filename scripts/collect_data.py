import requests
import re
import unicodedata

# Urls to the philosophy texts
urls = {
    "meditations": "https://www.gutenberg.org/cache/epub/2680/pg2680.txt",
    "enchiridion": "https://www.gutenberg.org/cache/epub/45109/pg45109.txt",
    "seneca_morals": "https://www.gutenberg.org/cache/epub/56075/pg56075.txt",
    "nicomachean_ethics": "https://www.gutenberg.org/cache/epub/8438/pg8438.txt",
    "the_republic": "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",
    "beyond_good_and_evil": "https://www.gutenberg.org/cache/epub/4363/pg4363.txt",
    "art_of_war": "https://www.gutenberg.org/cache/epub/17405/pg17405.txt",
    "tusculan": "https://www.gutenberg.org/cache/epub/14988/pg14988.txt",
    "bacon_essays": "https://www.gutenberg.org/cache/epub/575/pg575.txt",
    "pragmatism": "https://www.gutenberg.org/cache/epub/5116/pg5116.txt",
    "discourse": "https://www.gutenberg.org/cache/epub/59/pg59.txt",
    "minor_dialogues": "https://www.gutenberg.org/cache/epub/64576/pg64576.txt",
    "first_series": "https://www.gutenberg.org/cache/epub/2944/pg2944.txt",
    "second_series": "https://www.gutenberg.org/cache/epub/2945/pg2945.txt",
    "politics": "https://www.gutenberg.org/cache/epub/6762/pg6762.txt",
    "consolation": "https://www.gutenberg.org/cache/epub/14328/pg14328.txt",
    "walden": "https://www.gutenberg.org/cache/epub/205/pg205.txt",
    "selection_from_discourses": "https://www.gutenberg.org/cache/epub/10661/pg10661.txt",
    "conduct_of_life": "https://www.gutenberg.org/cache/epub/39827/pg39827.txt",
    "maxims": "https://www.gutenberg.org/cache/epub/9105/pg9105.txt",
    "on_friendship": "https://www.gutenberg.org/cache/epub/2808/pg2808.txt",
    "nature": "https://www.gutenberg.org/cache/epub/29433/pg29433.txt",
    "essays_of_schop": "https://www.gutenberg.org/cache/epub/10741/pg10741.txt",
    "praise_of_folly": "https://www.gutenberg.org/cache/epub/30201/pg30201.txt",
    "dawn_of_day": "https://www.gutenberg.org/cache/epub/39955/pg39955.txt",
    "essays": "https://www.gutenberg.org/cache/epub/36120/pg36120.txt",
    "candide": "https://www.gutenberg.org/cache/epub/19942/pg19942.txt",
    "essays_of_montaigne": "https://www.gutenberg.org/cache/epub/3600/pg3600.txt",
}

# Where each book's philosophy text actually start and end
boundaries = {
    "meditations":  ("Of my grandfather Verus", "Go thy ways then well pleased"),
    "enchiridion":  ("There are things which are within our power", "Thus Socrates became perfect"),
    "seneca_morals": ("It is, perhaps, one of the most pernicious", "not to oppress them"),
    "nicomachean_ethics": ("Every art, and every science reduced", "regulations are best for each"),
    "the_republic": ("I went down yesterday to the Piraeus with Glaucon", "in this life and in the pilgrimage of a thousand"),
    "beyond_good_and_evil": ("The Will to Truth, which is to tempt us to many a hazardous", "marvels of my solitude, you, my old, beloved--EVIL thoughts"),
    "art_of_war": ("The art of war is of vital importance to the State.","Spies are a most important"),
    "tusculan": ("At a time when I had entirely, or to a great degree,","suddenly revives out of the most desperate"),
    "bacon_essays": ("WHAT is truth? said jesting Pilate","have as great a watch"),
    "pragmatism": ("In the preface to that admirable collection","type of theism is exactly what you require"),
    "discourse": ("Good sense is, of all things among men","who might offer me the highest"),
    "minor_dialogues": ("You have asked me, Lucilius","mothers, they would have"),
    "first_series": ("There is no great and no small","continuations of the material creation"),
    "second_series": ("A moody child and wildly","the past?"),
    "politics": ("As we see that every city is a society,","education, moderation, possibility, and decorum."),
    "consolation": ("Who wrought my studious numbers","eyes of a Judge who seeth all things."),
    "walden": ("When I wrote the following pages","which also I have imagined, but not yet"),
    "selection_from_discourses": ("OF THE THINGS WHICH ARE IN OUR","the burden, by taking a part of it."),
    "conduct_of_life": ("It chanced during one winter,","they alone with him alone."),
    "maxims": ("What we term virtue is often but a mass","proportion they are removed from that point."),
    "on_friendship": ("Let this, then, be laid down as the first law","all things is Friendship"),
    "nature": ("TO go into solitude","sight."),
    "essays_of_schop": ("The differences which come under the first head","the poet and the philosopher!"),
    "praise_of_folly": ("slightly soever I am esteemed","t was spoke by Folly"),
    "dawn_of_day": ("SUBSEQUENT JUDGMENT.—All things that endure","brethren? or—?"),
    "essays": ("Some people are subject to a certain","the other from modern times."),
    "candide": ("In a castle of Westphalia,","let us cultivate our"),
    "essays_of_montaigne": ("It is one of the most conspicuous follies","subterranean manners to be of"),
}

# Get the raw_text text within the range we need
def extract_content(text, start_text, end_text):
    
    # Get the indexes of the first and last text that we actually want
    start_index = text.find(start_text)
    end_index = text.find(end_text)
    
    # display error message if start or end index could not be found
    if start_index == -1:
        raise ValueError(f"Start boundary not found: {start_text[:50]}")
    if end_index == -1:
        raise ValueError(f"End boundary not found: {end_text[:50]}")
    
    # Get all the text from the beginning of the start index to the end of the end index
    return text[start_index:end_index + len(end_text)]


def clean_text(text):
    
    # Replace all occurences of the windows new line representation to make everything consistent
    text = text.replace('\r\n', '\n')

    # Normalize unicode so smart quotes and dashes become predictable before cleaning
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", "-").replace("–", "-")
    
    # Remove any remaining non ASCII characters after normalization
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Remove anything that is a set of brackets and any characters within
    text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
    
    # Remove curly braced footnotes
    text = re.sub(r'\{[^}]*\}', '', text)
    
    # Remove leading ellipses
    text = re.sub(r'^\s*(\.\s*){2,}', '', text, flags=re.MULTILINE)
    
    # Clean up lines that start with connective punctuation
    text = re.sub(r'^\s*[,;:]\s+', '', text, flags=re.MULTILINE)
    
    # Remove underscores
    text = re.sub(r'_+', '', text)
    
    # Clean up the etc. formatting seen in the text
    text = text.replace('&c.', 'etc.')

    # Remove lines that start with a roman numeral and consist of spaces then capital letters and then a period
    text = re.sub(r'^[IVXLCDM]+\.\s+[A-Z\s]+\.$', '', text, flags=re.MULTILINE)
    
    # Remove lines that start with CHAPTER and are followed by a roman numeral
    text = re.sub(r'^CHAPTER\s+[IVXLCDM]+\.\s*$', '', text, flags=re.MULTILINE)
    
    # Remove lines that start with chapter and have a title after the roman numerals
    text = re.sub(r'^CHAPTER\s+[IVXLCDM]+\.\s+[A-Z\s]+$', '', text, flags=re.MULTILINE)

    # Remove chapter headers written as CHAP. II / BOOK II / PART II
    text = re.sub(r'^\s*CHAP\.\s+[IVXLCDM]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*BOOK\s+[IVXLCDM]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*PART\s+[IVXLCDM]+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove lines that start with not all caps Chapter and are followed by a roman numeral
    text = re.sub(r'^Chapter\s+[IVXLCDM]+\.\s*$', '', text, flags=re.MULTILINE)
    
    # Remove all-caps section headers and table-of-contents style headings
    text = re.sub(r'^\s*[A-Z][A-Z\s,\-;:\'\"\?\!\.]{8,}\s*$', '', text, flags=re.MULTILINE)
    
    # Remove roman numeral lines that are by themselves
    text = re.sub(r'^[IVXLCDM]+\.\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[IVXLCDM]+\s*$', '', text, flags=re.MULTILINE)

    # Remove digits that are by themselves
    text = re.sub(r'^\d+\.\s*$', '', text, flags=re.MULTILINE)
    
    # Remove roman numeral prefixes from beginning of lines
    text = re.sub(r'^[IVXLCDM]+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove number prefixes from beginning of lines
    text = re.sub(r'^[0-9]+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[0-9]+\.\-+\s*', '', text, flags=re.MULTILINE)
    
    # Remove lines that are just asterisks and spaces
    text = re.sub(r'^\s*\*[\s\*]*$', '', text, flags=re.MULTILINE)

    # Remove simple dialogue speaker labels like "A." or "M." at the start of a line
    text = re.sub(r'^\s*[A-Z]\.\s+', '', text, flags=re.MULTILINE)

    # Remove extra spaces before punctuation and collapse repeated spaces
    text = re.sub(r'\s+([,;:\.\?!])', r'\1', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    
    # Replace single newlines with a space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

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
    
