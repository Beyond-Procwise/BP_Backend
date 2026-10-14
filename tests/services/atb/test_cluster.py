from src.services.atb.pptx_import.cluster import EMPTY_ROW, group, rows_of, signature
from src.services.atb.pptx_import.read import Box, Deck, Shape


def _shape(x, y, w=2.0, h=1.0, kind='text', slide=1, table=None):
    return Shape(kind=kind, box=Box(x, y, w, h), runs=(), fill=None, line=None,
                 table=table, slide=slide, name='s')


def _deck(slides, height_in=7.5):
    return Deck(width_in=13.333, height_in=height_in, slides=tuple(slides),
                theme={'colours': {}, 'fonts': {}}, chart_series_colours=(), run_langs={})


def test_the_title_band_and_the_footer_are_not_part_of_the_signature():
    shapes = (_shape(0.5, 0.35, 12.33, 0.6), _shape(0.5, 2.0), _shape(0.5, 7.02, 11.7, 0.3))
    assert signature(shapes, 7.5) == ((1, 'SHAPE'),)


def test_a_decoration_is_not_a_column():
    shapes = (_shape(0.5, 2.0), _shape(4.0, 2.0), _shape(8.0, 2.05, 0.3, 0.3))
    assert signature(shapes, 7.5) == ((2, 'SHAPE'),)


def test_columns_within_a_tenth_of_an_inch_are_one_column():
    shapes = (_shape(0.5, 2.0), _shape(0.55, 2.0), _shape(4.0, 2.0))
    assert signature(shapes, 7.5) == ((2, 'SHAPE'),)


def test_a_table_row_and_a_chart_row_never_merge():
    table = signature((_shape(0.5, 2.0, kind='table'),), 7.5)
    chart = signature((_shape(0.5, 2.0, kind='chart'),), 7.5)
    plain = signature((_shape(0.5, 2.0),), 7.5)
    assert table != chart and table != plain and chart != plain


def test_a_table_anywhere_in_a_row_makes_it_a_table_row():
    mixed = (_shape(0.5, 2.0), _shape(4.0, 2.0, kind='table'))
    assert signature(mixed, 7.5) == ((2, 'TABLE'),)


def test_a_trailing_repeated_row_collapses_to_one_marked_repeating():
    rows = tuple(_shape(0.5 + i * 3, y) for y in (2.0, 3.5, 5.0) for i in range(4))
    sig = signature(rows, 7.5)
    assert len(sig) == 1
    assert sig[0] == (4, 'SHAPE', 'repeat')


def test_two_different_rows_do_not_collapse():
    shapes = (_shape(0.5, 2.0), _shape(0.5, 3.5), _shape(4.0, 3.5))
    assert signature(shapes, 7.5) == ((1, 'SHAPE'), (2, 'SHAPE'))


def test_a_slide_with_nothing_in_its_body_is_marked_empty():
    assert signature((_shape(0.5, 0.35, 12.33, 0.6),), 7.5) == (EMPTY_ROW,)


def test_rows_are_grouped_by_top_edge_within_a_tolerance():
    shapes = (_shape(0.5, 2.0), _shape(4.0, 2.3), _shape(0.5, 3.5))
    assert [len(r) for r in rows_of(shapes, 7.5)] == [2, 1]


def test_groups_slides_by_signature_and_orders_by_use():
    # The SINGLE-USE structure is on slide 1 and the reused one on 2, 3 and 4, so ordering by use
    # and ordering by first slide disagree. With it the other way round both orderings look the
    # same and the test proves nothing.
    once = (_shape(0.5, 2.0, kind='table'),)
    thrice = (_shape(0.5, 2.0), _shape(4.0, 2.0))
    clusters = group(_deck([once, thrice, thrice, thrice]))
    assert [len(c.slides) for c in clusters] == [3, 1]
    assert clusters[0].slides == (2, 3, 4)
    assert clusters[0].reused is True
    assert clusters[1].slides == (1,)
    assert clusters[1].reused is False


def test_a_cluster_carries_every_member_s_rows_not_just_the_first_s():
    # Without this a region's box could only be derived from one slide, and "this member is off
    # the median" had nothing to compare across. Every test that builds a Cluster by hand passes
    # rows_by_slide itself, so only this one exercises group() filling it in.
    first = (_shape(0.5, 2.0, 12.0, 1.0, slide=1),)
    second = (_shape(0.5, 2.0, 11.0, 1.0, slide=2),)
    clusters = group(_deck([first, second]))
    assert len(clusters) == 1
    cluster = clusters[0]
    assert len(cluster.rows_by_slide) == 2
    assert [rows[0][0].box.w for rows in cluster.rows_by_slide] == [12.0, 11.0]
    assert cluster.rows == cluster.rows_by_slide[0]


def test_the_footer_band_follows_the_page_height():
    # On a portrait page 7.02in is in the BODY, not the footer.
    shapes = (_shape(0.5, 7.02, 11.7, 0.3),)
    assert signature(shapes, 11.69) == ((1, 'SHAPE'),)
    assert signature(shapes, 7.5) == (EMPTY_ROW,)
