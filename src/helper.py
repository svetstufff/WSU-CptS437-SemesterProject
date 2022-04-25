import mpld3

def clear_last_line():
    print ("\033[A                             \033[A")

def show_progress(i, n):
    i += 1
    if (i > 1):
        clear_last_line()
    print("\t", round(100*(i / n), 1), "%", sep="")

def save_graph(fig, filename):
    html_str = mpld3.fig_to_html(fig)
    Html_file = open(f'graphs/{filename}.html',"w")
    Html_file.write(html_str)
    Html_file.close()