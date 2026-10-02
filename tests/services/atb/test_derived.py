from src.services.atb.pptx_import.derived import rating_scales, series_palette, writing
from src.services.atb.pptx_import.evidence import Evidence
from src.services.atb.pptx_import.read import Box, Deck, Run, Shape


def _deck(**kw):
    base = dict(width_in=13.333, height_in=7.5, slides=(), theme={'colours': {}, 'fonts': {}},
                chart_series_colours=(), run_langs={})
    base.update(kw)
    return Deck(**base)


def _text_deck(words, lang='en-US'):
    runs = tuple(Run(text=w, size_pt=12, font='Calibri', colour='#172033', bold=False, lang=lang)
                 for w in words)
    shape = Shape(kind='text', box=Box(0.5, 1.5, 6, 1), runs=runs, fill=None, line=None,
                  table=None, slide=1, name='TextBox 1')
    return _deck(slides=((shape,),), run_langs={lang: len(words)})


def test_the_series_palette_comes_from_the_charts_in_first_use_order():
    ev = Evidence()
    deck = _deck(chart_series_colours=('#0F6E78', '#B42D2D', '#172033'))
    assert series_palette(deck, ev, {'ink': '#172033', 'accent': '#2350C8'}) == \
        ['#0F6E78', '#B42D2D', '#172033']
    assert ev.as_dict()['values']['series_palette']['charts'] is True


def test_a_deck_with_no_charts_still_gets_a_non_empty_palette():
    # Review Focus 3: series_palette must be non-empty or the pack cannot validate.
    ev = Evidence()
    out = series_palette(_deck(), ev, {'ink': '#172033', 'accent': '#2350C8',
                                      'accent_2': '#0F6E78'})
    assert out, 'a pack with an empty series palette fails the contract'
    assert out[0] == '#2350C8', 'the accent leads when the deck has no charts'
    assert ev.as_dict()['values']['series_palette']['charts'] is False


def test_a_deck_with_neither_charts_nor_accents_falls_back_to_the_ink():
    ev = Evidence()
    assert series_palette(_deck(), ev, {'ink': '#172033'}) == ['#172033']


def test_a_rating_scale_needs_both_a_vocabulary_and_fills():
    ev = Evidence()
    rows = (('Area', 'Risk'), ('Freight', 'Low'), ('IT', 'High'), ('Tail', 'Low'))
    table = Shape(kind='table', box=Box(0.5, 1.5, 12.33, 4.0), runs=(), fill=None, line=None,
                  table=rows, slide=1, name='Table 1')
    out = rating_scales(_deck(slides=((table,),)), ev)
    assert out == {}, 'no cell fill means no chip colour, so no scale'
    flagged = [i for i in ev.as_dict()['incidental'] if i['kind'] == 'rating_scale']
    assert len(flagged) == 1, 'only Risk repeats a vocabulary; Area is a label column'
    assert 'Risk' in flagged[0]['value']
    assert 'no cell fill' in flagged[0]['why']
    assert 'defined by hand' in flagged[0]['why']


def test_a_column_of_free_prose_is_not_a_rating_candidate():
    ev = Evidence()
    rows = (('Area', 'Effect'),
            ('Freight', 'One rate card across four carriers'),
            ('IT', 'Re-tender before the auto-renewal date'),
            ('Tail', 'Catalogue the 1,400 smallest suppliers'))
    table = Shape(kind='table', box=Box(0.5, 1.5, 12.33, 4.0), runs=(), fill=None, line=None,
                  table=rows, slide=1, name='Table 1')
    rating_scales(_deck(slides=((table,),)), ev)
    assert not [i for i in ev.as_dict()['incidental'] if i['kind'] == 'rating_scale']


def test_a_repeated_SENTENCE_is_not_a_rating_vocabulary():
    # The repetition rule alone would accept this: two distinct values over four rows. A rating
    # chip is a word, not a clause, which is what the length limit is for.
    ev = Evidence()
    long_a = 'Dedicated strategy and senior sponsor'
    long_b = 'Compete hard and use the volume'
    rows = (('Area', 'What it means'), ('Cloud', long_a), ('Compute', long_b),
            ('Security', long_a), ('Tail', long_b))
    table = Shape(kind='table', box=Box(0.5, 1.5, 12.33, 4.0), runs=(), fill=None, line=None,
                  table=rows, slide=1, name='Table 1')
    rating_scales(_deck(slides=((table,),)), ev)
    assert not [i for i in ev.as_dict()['incidental'] if i['kind'] == 'rating_scale']


def test_two_british_spellings_in_a_whole_deck_do_not_contest_anything():
    # One or two could be a quotation or a supplier's name. The floor is three.
    ev = Evidence()
    out = writing(_text_deck(['Virtualisation', 'utilisation', 'and', 'nothing', 'else']), ev)
    assert out['locale_contested'] is False


def test_a_deck_that_declares_en_gb_is_never_contested():
    # There is nothing to contest: the declaration and the spelling already agree.
    ev = Evidence()
    out = writing(_text_deck(['Virtualisation', 'Mobilise', 'Optimise', 'utilisation'],
                             lang='en-GB'), ev)
    assert out['locale'] == 'en-GB'
    assert out['locale_contested'] is False


def test_the_declared_language_is_recorded_and_contested_by_the_spelling():
    ev = Evidence()
    out = writing(_text_deck(['Virtualisation', 'Mobilise', 'Optimise', 'utilisation']), ev)
    assert out['locale'] == 'en-US'
    assert out['locale_contested'] is True
    assert out['locale_suggested'] == 'en-GB'
    assert 'CONTESTED' in ev.as_dict()['values']['writing.locale']['why']


def test_an_american_deck_is_not_flagged():
    ev = Evidence()
    out = writing(_text_deck(['organize', 'the', 'color', 'center', 'optimization']), ev)
    assert out['locale_contested'] is False
    assert out['locale_suggested'] == 'en-US'


def test_words_that_merely_look_british_do_not_contest_anything():
    # four, hour, your, otherwise, promise, exercise: all end in -our or -ise and none of them is
    # a British spelling of anything.
    ev = Evidence()
    out = writing(_text_deck(['four', 'hour', 'your', 'otherwise', 'promise', 'exercise',
                              'advertise', 'franchise']), ev)
    assert out['locale_contested'] is False


def test_a_deck_that_declares_no_language_defaults_to_en_gb():
    ev = Evidence()
    out = writing(_deck(), ev)
    assert out['locale'] == 'en-GB'
    assert out['locale_contested'] is False
