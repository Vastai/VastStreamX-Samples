#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import vaststreamx as vsx
import numpy as np

if __name__ == "__main__":
    device_id = 0
    # init card
    assert vsx.set_device(device_id) == 0, f"set device_id {device_id} failed"

    # get card count
    card_count = vsx.get_all_cards_count()
    print(f"Find {card_count} cards in system.")

    # get all cards
    cards = vsx.get_all_cards()
    assert len(cards) > 0, "get_all_cards() return empty."

    # print card info
    for i, card in enumerate(cards):
        print(f"{i}th card info:")
        print(f"\tUUID: {card.uuid}")
        print(f"\tCard type:: {card.card_type}")
        info = "\tDie ID: "
        for die in card.dies:
            info += f"{die.device_id}, "
        print(info)

    # get die0 status
    Byte_to_GB = 1.0 / (1024.0 * 1024.0 * 1024.0)
    die = vsx.Die(device_id)
    print(f"Device id {device_id} status: ")
    print(f"\tTemperature: {die.temperature['temperature'][0] * 0.01}  ℃.")
    print(f"\tPower: {die.power['power'][3] * 0.000001} W.")
    print(f"\tMemory total: {die.memory['total'] * Byte_to_GB:.3f} GB.")
    print(f"\tMemory free: {die.memory['free'] * Byte_to_GB:.3f} GB.")
    print(f"\tMemory used: {die.memory['used'] * Byte_to_GB:.3f} GB.")
    print(f"\tMemory usage rate: {die.memory['usage_rate']:.2f}%.")
    print(f"\tAI usage rate: {die.utilization['ai']:.2f}%.")
    use_rate = np.mean(die.utilization["vdsp"])
    print(f"\tVDSP usage rate: {use_rate:.2f}%.")
    use_rate = np.mean(die.utilization["vdmcu"])
    print(f"\tDEC usage rate: {use_rate:.2f}%.")
    use_rate = np.mean(die.utilization["vemcu"])
    print(f"\tNEC usage rate: {use_rate:.2f}%.")
