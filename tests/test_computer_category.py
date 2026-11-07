from gs_search.perception import category_identity


def test_computer_category_covers_desktop_and_laptop_without_assuming_monitor_is_pc():
    for text in ['computer','desktop computer','PC','desktop PC','laptop','laptop computer','notebook computer']:
        assert category_identity(text,'computer')=='computer'
    for text in ['monitor','television','tablet','notebook','chair']:
        assert category_identity(text,'computer')!='computer'
