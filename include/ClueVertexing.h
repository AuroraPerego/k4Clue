/*
 * Copyright (c) 2020-2024 Key4hep-Project.
 *
 * This file is part of Key4hep.
 * See https://key4hep.github.io/key4hep-doc/ for further info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef CLUE_VERTEXING_H
#define CLUE_VERTEXING_H

#include "Gaudi/Property.h"
#include "GaudiKernel/ITHistSvc.h"
#include "k4FWCore/Consumer.h"

#include <edm4hep/ReconstructedParticleCollection.h>
#include <edm4hep/VertexCollection.h>

#include "ClueBackend.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "TH1F.h"
#include "TTree.h"

using VertexColl = edm4hep::VertexCollection;
using PartColl = edm4hep::ReconstructedParticleCollection;

/// One reconstructed track's kinematic/timing info, local to a particle.
struct TrackInfo {
  float D0, phi, omega, Z0, tanLambda;
  float refX, refY, refZ;
  float zip;      // Z0 at IP-like state, used as CLUE coordinate
  float pt;
  float time;     // raw hit time
  float t0;       // propagated time at target (IP or vertex)
  float path;
  float beta;
};

/// One cluster's info, local to a particle.
struct ClusterInfo {
  float energy, x, y, z, time;
  float propTime, propPath;
  std::vector<float> hitsX, hitsY, hitsZ, hitsTime;
};

/// Per-particle summary used to build/refill vertex info.
struct ParticleInfo {
  bool hasTrack = false;
  bool usedInVertex = false;
  TrackInfo track;                 // valid only if hasTrack
  std::vector<ClusterInfo> clusters;
  float particleTime = -99.f;      // averaged, propagated at vertex
};

struct CLUEVertexing final : k4FWCore::Consumer<void(const VertexColl&, const PartColl&)> {
  CLUEVertexing(const std::string& name, ISvcLocator* svcLoc)
      : Consumer(name, svcLoc,
                 {KeyValues("VertexColl", {"PrimaryVertices"}),
                  KeyValues("RecoParticles", {"PandoraPFOs"})}) {}

  StatusCode initialize() override;
  StatusCode finalize() override;

  ~CLUEVertexing();

  void operator()(const VertexColl& vtx_coll, const PartColl& part_coll) const override;

  // ---- setup helpers ----
  void initializeTrees();
  void cleanTrees() const;

  // ---- CLUE vertexing ----
  /// Runs CLUE on (zip, pt) pairs to find candidate vertex clusters.
  std::vector<int> runClueVertexing(const std::vector<float>& zip,
                                     const std::vector<float>& pt) const;

  // ---- per-particle extraction ----
  /// Fills TrackInfo from the first available track, propagated to `target`.
  /// Returns false if the particle has no tracks (target propagation skipped).
  bool fillTrackInfo(const edm4hep::ReconstructedParticle& part,
                      const edm4hep::Vector3f& target, TrackInfo& out) const;

  /// Fills ClusterInfo for all clusters of a particle.
  /// If hasTrack is true, propagates the track to each cluster position;
  /// otherwise infers a direction (PCA over cluster hits, or straight to origin)
  /// and propagates that pseudo-track instead.
  std::vector<ClusterInfo> fillClusterInfo(const edm4hep::ReconstructedParticle& part,
                                            bool hasTrack, const TrackInfo& trackAtLastHit) const;

  /// PCA-based direction estimate from a set of hit positions.
  edm4hep::Vector3f estimateDirectionPCA(const std::vector<float>& x,
                                          const std::vector<float>& y,
                                          const std::vector<float>& z) const;

  /// Builds a straight-line "virtual track" time propagation from a cluster
  /// back towards the origin (or along the PCA direction), given a beta.
  std::pair<float, float> propagateClusterTime(const edm4hep::Vector3f& clusterPos,
                                                const edm4hep::Vector3f& direction,
                                                float beta = 1.0f) const;

  /// Combines all timing info for a particle into a single propagated time.
  float computeParticleTime(const ParticleInfo& p) const;

private:
  SmartIF<ITHistSvc> m_ths;

  // CLUE algo
  mutable ClueBackend<1>* m_backend = nullptr;

  // Vertex-level
  mutable std::vector<float> v_x, v_y, v_z, v_time, v_timeErr;
  mutable std::vector<std::vector<int>> v_particles;

  // Particle-level
  mutable std::vector<std::vector<int>> part_tracks, part_clusters;
  mutable std::vector<bool> p_isVertexAssociated;
  mutable std::vector<float> p_time;

  // Track-level
  mutable std::vector<float> track_time, track_t0, track_path, track_beta;
  mutable std::vector<float> trk_D0, trk_phi, trk_omega, trk_Z0, trk_tanLambda;
  mutable std::vector<float> trk_refPointX, trk_refPointY, trk_refPointZ;
  mutable std::vector<float> trk_zip, trk_pt;

  // Cluster-level
  mutable std::vector<float> clus_energy, clus_x, clus_y, clus_z, clus_time;
  mutable std::vector<float> clus_propTime, clus_propPath;
  mutable std::vector<std::vector<float>> clus_hits_x, clus_hits_y, clus_hits_z, clus_hits_time;

  mutable TTree* t_vertices = nullptr;
  mutable TTree* t_particles = nullptr;
  mutable TTree* t_tracks = nullptr;
  mutable TTree* t_clusters = nullptr;

  Gaudi::Property<float> m_dc{this, "dc", 30.f};
  Gaudi::Property<float> m_rhoc{this, "rhoc", 0.1f};
  Gaudi::Property<float> m_dm{this, "dm", 30.f};
  Gaudi::Property<float> m_seed_dc{this, "SeedCriticalDistance", 10};
  Gaudi::Property<int>   m_pointsPerBin{this, "PointsPerBin", 10};
};

#endif // CLUE_VERTEXING_H
