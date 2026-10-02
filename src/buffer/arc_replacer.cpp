// :bustub-keep-private:
//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// arc_replacer.cpp
//
// Identification: src/buffer/arc_replacer.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "buffer/arc_replacer.h"
#include <optional>
#include "common/config.h"
#include <algorithm>

namespace bustub {

/**
 *
 * TODO(P1): Add implementation
 *
 * @brief a new ArcReplacer, with lists initialized to be empty and target size to 0
 * @param num_frames the maximum number of frames the ArcReplacer will be required to cache
 */
ArcReplacer::ArcReplacer(size_t num_frames) : replacer_size_(num_frames) {}

/**
 * TODO(P1): Add implementation
 *
 * @brief Performs the Replace operation as described by the writeup
 * that evicts from either mfu_ or mru_ into its corresponding ghost list
 * according to balancing policy.
 *
 * If you wish to refer to the original ARC paper, please note that there are
 * two changes in our implementation:
 * 1. When the size of mru_ equals the target size, we don't check
 * the last access as the paper did when deciding which list to evict from.
 * This is fine since the original decision is stated to be arbitrary.
 * 2. Entries that are not evictable are skipped. If all entries from the desired side
 * (mru_ / mfu_) are pinned, we instead try victimize the other side (mfu_ / mru_),
 * and move it to its corresponding ghost list (mfu_ghost_ / mru_ghost_).
 *
 * @return frame id of the evicted frame, or std::nullopt if cannot evict
 */
auto ArcReplacer::Evict() -> std::optional<frame_id_t> {
  std::lock_guard<std::mutex> lock(latch_);

  if (curr_size_ == 0) {
    return std::nullopt;
  }

  // Search from the oldest end, skipping non-evictable frames.
  auto evict_from = [&](std::list<frame_id_t> &resident,
                        std::list<page_id_t> &ghost,
                        ArcStatus ghost_status)
      -> std::optional<frame_id_t> {
    auto position = resident.end();

    while (position != resident.begin()) {
      --position;

      const frame_id_t frame_id = *position;
      auto entry = alive_map_.at(frame_id);

      if (!entry->evictable_) {
        continue;
      }

      const page_id_t page_id = entry->page_id_;

      resident.erase(position);
      alive_positions_.erase(frame_id);
      alive_map_.erase(frame_id);

      // Retain the page's history, but detach it from the frame.
      entry->frame_id_ = static_cast<frame_id_t>(-1);
      entry->evictable_ = false;
      entry->arc_status_ = ghost_status;

      ghost.push_front(page_id);
      ghost_map_[page_id] = entry;
      ghost_positions_[page_id] = ghost.begin();

      --curr_size_;
      return frame_id;
    }

    return std::nullopt;
  };

  if (mru_.size() < mru_target_size_) {
    auto victim = evict_from(mfu_, mfu_ghost_, ArcStatus::MFU_GHOST);
    if (victim.has_value()) {
      return victim;
    }
    return evict_from(mru_, mru_ghost_, ArcStatus::MRU_GHOST);
  }

  auto victim = evict_from(mru_, mru_ghost_, ArcStatus::MRU_GHOST);
  if (victim.has_value()) {
    return victim;
  }
  return evict_from(mfu_, mfu_ghost_, ArcStatus::MFU_GHOST);
}

/**
 * TODO(P1): Add implementation
 *
 * @brief Record access to a frame, adjusting ARC bookkeeping accordingly
 * by bring the accessed page to the front of mfu_ if it exists in any of the lists
 * or the front of mru_ if it does not.
 *
 * Performs the operations EXCEPT REPLACE described in original paper, which is
 * handled by `Evict()`.
 *
 * Consider the following four cases, handle accordingly:
 * 1. Access hits mru_ or mfu_
 * 2/3. Access hits mru_ghost_ / mfu_ghost_
 * 4. Access misses all the lists
 *
 * This routine performs all changes to the four lists as preperation
 * for `Evict()` to simply find and evict a victim into ghost lists.
 *
 * Note that frame_id is used as identifier for alive pages and
 * page_id is used as identifier for the ghost pages, since page_id is
 * the unique identifier to the page after it's dead.
 * Using page_id for alive pages should be the same since it's one to one mapping,
 * but using frame_id is slightly more intuitive.
 *
 * @param frame_id id of frame that received a new access.
 * @param page_id id of page that is mapped to the frame.
 * @param access_type type of access that was received. This parameter is only needed for
 * leaderboard tests.
 */
void ArcReplacer::RecordAccess(frame_id_t frame_id, page_id_t page_id,
                               [[maybe_unused]] AccessType access_type) {
  std::lock_guard<std::mutex> lock(latch_);

  BUSTUB_ENSURE(frame_id >= 0, "Invalid frame id");

  // Case 1: the frame is already resident.
  auto alive = alive_map_.find(frame_id);
  if (alive != alive_map_.end()) {
    auto entry = alive->second;

    BUSTUB_ENSURE(entry->page_id_ == page_id,
                  "Resident frame cannot change page without removal");

    auto position = alive_positions_.at(frame_id);

    if (entry->arc_status_ == ArcStatus::MRU) {
      mfu_.splice(mfu_.begin(), mru_, position);
    } else {
      mfu_.splice(mfu_.begin(), mfu_, position);
    }

    entry->arc_status_ = ArcStatus::MFU;
    alive_positions_[frame_id] = mfu_.begin();
    return;
  }

  // Cases 2/3: the page has ghost history.
  auto ghost = ghost_map_.find(page_id);
  if (ghost != ghost_map_.end()) {
    auto entry = ghost->second;

    // Calculate the adjustment before removing the ghost entry.
    if (entry->arc_status_ == ArcStatus::MRU_GHOST) {
      const size_t delta =
          std::max(size_t{1}, mfu_ghost_.size() / mru_ghost_.size());

      mru_target_size_ +=
          std::min(delta, replacer_size_ - mru_target_size_);

      mru_ghost_.erase(ghost_positions_.at(page_id));
    } else {
      const size_t delta =
          std::max(size_t{1}, mru_ghost_.size() / mfu_ghost_.size());

      mru_target_size_ -= std::min(delta, mru_target_size_);

      mfu_ghost_.erase(ghost_positions_.at(page_id));
    }

    ghost_positions_.erase(page_id);
    ghost_map_.erase(ghost);

    entry->frame_id_ = frame_id;
    entry->evictable_ = false;
    entry->arc_status_ = ArcStatus::MFU;

    mfu_.push_front(frame_id);
    alive_map_[frame_id] = entry;
    alive_positions_[frame_id] = mfu_.begin();
    return;
  }

  // Case 4: a new page with no retained history.
  auto forget_oldest = [&](std::list<page_id_t> &list) {
    BUSTUB_ENSURE(!list.empty(), "Expected a ghost entry to discard");

    const page_id_t old_page_id = list.back();
    list.pop_back();
    ghost_positions_.erase(old_page_id);
    ghost_map_.erase(old_page_id);
  };

  if (mru_.size() + mru_ghost_.size() == replacer_size_) {
    forget_oldest(mru_ghost_);
  } else {
    const size_t total =
        mru_.size() + mfu_.size() + mru_ghost_.size() + mfu_ghost_.size();

    if (total == 2 * replacer_size_) {
      forget_oldest(mfu_ghost_);
    }
  }

  auto entry =
      std::make_shared<FrameStatus>(page_id, frame_id, false, ArcStatus::MRU);

  mru_.push_front(frame_id);
  alive_map_[frame_id] = entry;
  alive_positions_[frame_id] = mru_.begin();
}

/**
 * TODO(P1): Add implementation
 *
 * @brief Toggle whether a frame is evictable or non-evictable. This function also
 * controls replacer's size. Note that size is equal to number of evictable entries.
 *
 * If a frame was previously evictable and is to be set to non-evictable, then size should
 * decrement. If a frame was previously non-evictable and is to be set to evictable,
 * then size should increment.
 *
 * If frame id is invalid, throw an exception or abort the process.
 *
 * For other scenarios, this function should terminate without modifying anything.
 *
 * @param frame_id id of frame whose 'evictable' status will be modified
 * @param set_evictable whether the given frame is evictable or not
 */
void ArcReplacer::SetEvictable(frame_id_t frame_id, bool set_evictable) {
  std::lock_guard<std::mutex> lock(latch_);

  BUSTUB_ENSURE(frame_id >= 0, "Invalid frame id");

  auto found = alive_map_.find(frame_id);
  if (found == alive_map_.end()) {
    return;
  }

  auto &entry = *found->second;
  if (entry.evictable_ == set_evictable) {
    return;
  }

  entry.evictable_ = set_evictable;

  if (set_evictable) {
    ++curr_size_;
  } else {
    --curr_size_;
  }
}

/**
 * TODO(P1): Add implementation
 *
 * @brief Remove an evictable frame from replacer.
 * This function should also decrement replacer's size if removal is successful.
 *
 * Note that this is different from evicting a frame, which always remove the frame
 * decided by the ARC algorithm.
 *
 * If Remove is called on a non-evictable frame, throw an exception or abort the
 * process.
 *
 * If specified frame is not found, directly return from this function.
 *
 * @param frame_id id of frame to be removed
 */
void ArcReplacer::Remove(frame_id_t frame_id) {
  std::lock_guard<std::mutex> lock(latch_);

  auto found = alive_map_.find(frame_id);
  if (found == alive_map_.end()) {
    return;
  }

  auto entry = found->second;
  BUSTUB_ENSURE(entry->evictable_, "Cannot remove a non-evictable frame");

  auto position = alive_positions_.at(frame_id);

  if (entry->arc_status_ == ArcStatus::MRU) {
    mru_.erase(position);
  } else {
    mfu_.erase(position);
  }

  alive_positions_.erase(frame_id);
  alive_map_.erase(found);
  --curr_size_;

  // Explicit removal does not create ghost history.
}

/**
 * TODO(P1): Add implementation
 *
 * @brief Return replacer's size, which tracks the number of evictable frames.
 *
 * @return size_t
 */
auto ArcReplacer::Size() -> size_t { 
  std::lock_guard<std::mutex> lock(latch_);
  return curr_size_;
}

}  // namespace bustub
