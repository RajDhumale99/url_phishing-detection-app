from bs4 import BeautifulSoup
import os
import features as fe
import pandas as pd

# 1 Define a function that opens a HTML file and Returns the content
file_name = "mini_dataset/2.html"

def open_file(file_name):
    with open(file_name, "r", encoding='utf-8') as f:
        return f.read()

# 2 Define a function that creates a BeautifulSoup object
def create_soup(text):
    return BeautifulSoup(text, "html.parser")

# 3 Define a function that creates a Vector by running all feature functions for the soup object
def create_vector(soup):
    return [
        fe.has_title(soup),
        fe.has_input(soup),
        fe.has_button(soup),
        fe.has_image(soup),
        fe.has_submit(soup),
        fe.has_link(soup),
        fe.has_password(soup),
        fe.has_email_input(soup),
        fe.has_hidden_element(soup),
        fe.has_audio(soup),
        fe.has_video(soup),
        fe.number_of_inputs(soup),
        fe.number_of_buttons(soup),
        fe.number_of_images(soup),
        fe.number_of_option(soup),
        fe.number_of_list(soup),
        fe.number_of_TH(soup),
        fe.number_of_TR(soup),
        fe.number_of_href(soup),
        fe.number_of_paragraph(soup),
        fe.number_of_script(soup),
        fe.length_of_title(soup),
        fe.has_h1(soup),
        fe.has_h2(soup),
        fe.has_h3(soup),
        fe.length_of_text(soup),
        fe.number_of_clickable_button(soup),
        fe.number_of_a(soup),
        fe.number_of_img(soup),
        fe.number_of_div(soup),
        fe.number_of_figure(soup),
        fe.has_footer(soup),
        fe.has_form(soup),
        fe.has_text_area(soup),
        fe.has_iframe(soup),
        fe.has_text_input(soup),
        fe.number_of_meta(soup),
        fe.has_nav(soup),
        fe.has_object(soup),
        fe.has_picture(soup),
        fe.number_of_sources(soup),
        fe.number_of_span(soup),
        fe.number_of_table(soup)
    ]

# 4 Run step 1, 2, and 3 for all HTML files and create a 2-D array
folder = "mini_dataset"

def create_2d_list(folder_name):
    directory = os.path.join(os.getcwd(), folder_name)
    data = []
    for file in sorted(os.listdir(directory)):
        soup = create_soup(open_file(directory + "/" + file))
        data.append(create_vector(soup))
    return data


# 5 Create a DataFrame from the 2-D array
data = create_2d_list(folder)

columns = [
    "has_title",
    "has_input",
    "has_button",
    "has_image",
    "has_submit",
    "has_link",
    "has_password",
    "has_email_input",
    "has_hidden_element",
    "has_audio",
    "has_video",
    "number_of_inputs",
    "number_of_buttons",
    "number_of_images",
    "number_of_option",
    "number_of_list",
    "number_of_TH",
    "number_of_TR",
    "number_of_href",
    "number_of_paragraph",
    "number_of_script",
    "length_of_title",
    'has_h1',
    'has_h2',
    'has_h3',
    'length_of_text',
    'number_of_clickable_button',
    'number_of_a',
    'number_of_img',
    'number_of_div',
    'number_of_figure',
    'has_footer',
    'has_form',
    'has_text_area',
    'has_iframe',
    'has_text_input',
    'number_of_meta',
    'has_nav',
    'has_object',
    'has_picture',
    'number_of_sources',
    'number_of_span',
    'number_of_table'
]

df = pd.DataFrame(data=data, columns=columns)
print(df.head(5))
