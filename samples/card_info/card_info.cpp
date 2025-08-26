
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "glog/logging.h"
#include "vaststreamx/vaststreamx.h"

int main(int argc, char* argv[]) {
  // init card
  CHECK(vsx::SetDevice(0) == 0) << "SetDevice 0 failed";

  // get card count
  uint32_t card_count = 0;
  vsx::GetAllCardsCount(card_count);
  std::cout << "Find " << card_count << " cards in system.\n";

  // get all cards
  std::vector<vsx::Card> cards = vsx::GetAllCards();
  CHECK(cards.size() > 0) << "GetAllCards() return empty.";

  // print card info
  for (size_t i = 0; i < cards.size(); i++) {
    auto& card = cards[i];
    std::cout << i << "th card info:\n";
    std::string info;
    card.GetUUID(info);
    std::cout << "\tUUID: " << info << std::endl;
    card.GetCardType(info);
    std::cout << "\tCard type: " << info << std::endl;
    std::vector<vsx::Die> dies;
    card.GetAllDies(dies);
    std::cout << "\tDie ID: ";
    for (auto& die : dies) {
      uint32_t die_id;
      die.GetDeviceId(die_id);
      std::cout << die_id << ", ";
    }
    std::cout << std::endl;
  }

  // get die0 status
  vsx::Die die = vsx::Die(0);
  std::cout << "Device id 0 status: \n";

  vsx::DieTemperature temp;
  die.GetTemperature(temp);
  std::cout << "\tTemperature: " << temp.temperature[0] * 0.01 << " ℃.\n";

  vsx::DevicePower pow;
  die.GetPower(pow);
  std::cout << "\tPower: " << pow.power[3] * 0.000001 << " W.\n";

  double Byte_to_GB = 1.0 / (1024.0 * 1024.0 * 1024.0);
  vsx::DieMemory mem;
  die.GetMemory(mem);
  std::cout << "\tMemory total: " << mem.total * Byte_to_GB << " GB.\n";
  std::cout << "\tMemory free: " << mem.free * Byte_to_GB << " GB.\n";
  std::cout << "\tMemory used: " << mem.used * Byte_to_GB << " GB.\n";
  std::cout << "\tMemory usage rate: " << mem.usage_rate << "% .\n";

  vsx::DieUtilization util;
  die.GetUtilization(util);
  std::cout << "\tAI usage rate: " << util.ai << "% .\n";
  int len = sizeof(util.vdsp) / sizeof(util.vdsp[0]);
  float use_rate = 0;
  for (int i = 0; i < len; i++) {
    use_rate += util.vdsp[i];
  }
  use_rate /= len;
  std::cout << "\tVDSP usage rate: " << use_rate << "% .\n";
  len = sizeof(util.vdmcu) / sizeof(util.vdmcu[0]);
  use_rate = 0;
  for (int i = 0; i < len; i++) {
    use_rate += util.vdmcu[i];
  }
  use_rate /= len;
  std::cout << "\tDEC usage rate: " << use_rate << "% .\n";
  len = sizeof(util.vemcu) / sizeof(util.vemcu[0]);
  use_rate = 0;
  for (int i = 0; i < len; i++) {
    use_rate += util.vemcu[i];
  }
  use_rate /= len;
  std::cout << "\tENC usage rate: " << use_rate << "% .\n";
  return 0;
}
