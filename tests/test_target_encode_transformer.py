def test_future_struggle_outside_information(target_encoder):

    check = target_encoder.features_cv_map["x1"]["cv0"] == {
        "Pirates": 0,
        "Cubs": 0.5,
        "Reds": 0,
    }

    assert check


def test_past_success_outside_information(target_encoder):

    check = target_encoder.features_cv_map["x1"]["cv1"] == {
        "Pirates": 0.5,
        "Cubs": 1,
        "Reds": 0.5,
    }

    assert check


def test_average_past_future_cv(target_encoder):

    check = target_encoder.features_cv_map["x1"]["cv_mean"] == {
        "Pirates": 0.25,
        "Cubs": 0.75,
        "Reds": 0.25,
    }

    assert check


def test_transform_by_cv_fold(target_encoder, data_target_encode):

    xy_target_encode = data_target_encode.copy()

    xy_target_encode = target_encoder.transform(xy_target_encode)

    expected_encode_past_from_future = [0, 0, 0.5, 0.5, 0, 0]
    expected_encode_future_from_past = [0.5, 0.5, 1, 1, 0.5, 0.5]
    expected = (
        expected_encode_past_from_future + expected_encode_future_from_past
    )

    check = all(xy_target_encode["x1"] == expected)

    assert check
