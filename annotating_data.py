import pandas as pd


def save_progress(df, calibrate):
    # save progress to file
    out = df.to_json(orient='index', indent=4, force_ascii=False)
    if not calibrate:
        with open(f'data/annotated_reviews_lang_2.json', 'w', encoding='utf-8') as f:
            f.write(out)
    else:
        with open(f'data/calibration_reviews.json', 'w', encoding='utf-8') as f:
            f.write(out)
        pass



def start_annotate(calibrate=False):
    # load reviews
    df1 = pd.read_json(f'data/annotated_reviews_lang_2.json', orient='index')
    if calibrate:
        df2 = pd.read_json('data/annotated_reviews_lang_1.json', orient='index')
        df = pd.concat([df1.iloc[:125, :], df2.iloc[:125, :]])
        df.reset_index(drop=True, inplace=True)
    else:
        df = df1
    entities = ['amenities', 'location', 'cleanliness', 'staff']

    index = 0
    while (df.loc[index][-4:] != ['n/a']*4).all():
        index += 1
        if index == len(df.index):
            break

    while index < len(df.index):
        print(f"""{index}: {df['hotel_review'].loc[index]}""")
        options = ''
        while options not in ('c', 'd', 'exit', 'prev'):
            options = input('pick an option: c (continue), d (discard), prev (previous), exit ')
        if options == 'd':
            df.loc[index][-4:] = 'skipped'
            index += 1
            save_progress(df, calibrate)
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

        save_progress(df, calibrate)

        index += 1

# start_annotate(calibrate=True)
start_annotate()


