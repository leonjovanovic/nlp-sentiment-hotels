import pandas as pd

# load reviews
df = pd.read_json(f'hotels_reviews_annotated_0_1000.json', orient='index')
entities = ['price', 'location', 'cleanliness_room', 'staff']

index = 0
while (df.loc[index][-4:] != ['n/a']*4).all():
    index += 1

while index < len(df.index):
    print(f"""{index}: {df['hotel_review'].loc[index]}""")
    options = ''
    while options not in ('c', 'd', 'exit', 'prev'):
        options = input('pick an option: c (continue), d (discard), prev (previous), exit')
    if options == 'd':
        df.loc[index][-4:] = 'skipped'
        index += 1
        continue
    elif options == 'exit':
        break
    elif options == 'prev':
        index = index - 1 if index > 0 else index
        continue

    for i, entity in enumerate(entities):
        value = 'temp'
        while value not in ('p', 'n', ''):
            value = input(f'{entity}?')
        df.loc[index][-4+i] = value

    # save progress to file
    out = df.to_json(orient='index', indent=4, force_ascii=False)
    with open(f'hotels_reviews_annotated_0_1000.json', 'w', encoding='utf-8') as f:
        f.write(out)

    index += 1
