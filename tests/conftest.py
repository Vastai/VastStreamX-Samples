
# 
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--device_id", action="store", default="0", help="Device_id to run tests on"
    )

@pytest.fixture
def device_id(request):
    return request.config.getoption("--device_id")