import pandas as pd


def save_progress(df):
    # save progress to file
    out = df.to_json(orient='index', indent=4, force_ascii=False)
    with open(f'data/annotated_reviews_lang_1.json', 'w', encoding='utf-8') as f:
        f.write(out)


def start_annotate():
    # load reviews
    df = pd.read_json(f'data/annotated_reviews_lang_1.json', orient='index')
    entities = ['amenities', 'location', 'cleanliness', 'staff']

    index = 0
    while (df.loc[index][-4:] != ['n/a']*4).all():
        index += 1

    while index < len(df.index):
        print(f"""{index}: {df['hotel_review'].loc[index]}""")
        options = ''
        while options not in ('c', 'd', 'exit', 'prev'):
            options = input('pick an option: c (continue), d (discard), prev (previous), exit ')
        if options == 'd':
            df.loc[index][-4:] = 'skipped'
            index += 1
            save_progress(df)
            continue
        elif options == 'exit':
            break
        elif options == 'prev':
            index = index - 1 if index > 0 else index
            continue

        for i, entity in enumerate(entities):
            value = 'temp'
            while value not in ('p', 'n', ''):
                value = input(f'{entity}? ')
            df.loc[index][-4+i] = value

        save_progress(df)

        index += 1


start_annotate()

