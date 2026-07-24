/*
 * Copyright (c) 2020-2024 Key4hep-Project.
 * Licensed under the Apache License, Version 2.0.
 */
#include "ClueVertexing.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

#include "CLUEstering/CLUEstering.hpp"

DECLARE_COMPONENT(CLUEVertexing)

namespace {

constexpr float C_MM_PER_NS = 299.792458f;
constexpr float M_PION = 0.13957f; // GeV/c^2

enum class TrackStateLocation : int {
  AtOther = 0,
  AtIP = 1,
  AtFirstHit = 2,
  AtLastHit = 3,
  AtCalorimeter = 4,
  AtVertex = 5,
};

std::pair<float, float> computeHelixPathAndTimeToPoint(const edm4hep::TrackState& ts,
                                                         const edm4hep::Vector3f& target,
                                                         float beta = 1.0f) {
  if (beta <= 0) return {0.f, -99.f};

  const float D0 = ts.D0, phi0 = ts.phi, omega = ts.omega, tanL = ts.tanLambda;
  const float x1 = ts.referencePoint.x, y1 = ts.referencePoint.y, z1 = ts.referencePoint.z;
  const float x2 = target.x, y2 = target.y, z2 = target.z;

  if (std::fabs(omega) < 1e-9f) {
    const float dx = x2 - x1, dy = y2 - y1, dz = z2 - z1;
    const float pathLength = std::sqrt(dx * dx + dy * dy + dz * dz);
    return {pathLength, pathLength / (beta * C_MM_PER_NS)};
  }

  const float R = 1.0f / std::fabs(omega);
  const float xc = x1 + (D0 + R) * std::sin(phi0);
  const float yc = y1 - (D0 + R) * std::cos(phi0);

  const float phi1 = std::atan2(y1 - yc, x1 - xc);
  const float phi2 = std::atan2(y2 - yc, x2 - xc);

  float dphi = phi2 - phi1;
  if (dphi > M_PI) dphi -= 2.0f * M_PI;
  if (dphi < -M_PI) dphi += 2.0f * M_PI;

  const float pathLength = std::fabs(dphi / omega) * std::sqrt(1.f + tanL * tanL);
  return {pathLength, pathLength / (beta * C_MM_PER_NS)};
}

std::pair<float, float> computeMomentum(const edm4hep::TrackState& ts, float Bz = 2.0f) {
  constexpr float a = 3e-4f;
  if (std::fabs(ts.omega) < 1e-9f) return {0.f, 0.f};
  const float pT = a * std::fabs(Bz / ts.omega);
  const float p = pT * std::sqrt(1.0f + ts.tanLambda * ts.tanLambda);
  return {pT, p};
}

std::pair<float, float> computeWeightedMeanAndStd(const std::vector<float>& v,
                                                    const std::vector<float>& sigma) {
  std::vector<float> w(v.size());
  std::transform(sigma.begin(), sigma.end(), w.begin(), [](float s) { return 1.f / (s * s); });
  const float wsum = std::accumulate(w.begin(), w.end(), 0.f);
  const float wmean = std::inner_product(v.begin(), v.end(), w.begin(), 0.f) / wsum;
  return {wmean, std::sqrt(1.f / wsum)};
}

} // namespace


CLUEVertexing::~CLUEVertexing() {
  if (m_backend != nullptr) {
    destroyBackend<1>(m_backend);
    m_backend = nullptr;
  }
}

StatusCode CLUEVertexing::initialize() {
  if (Gaudi::Algorithm::initialize().isFailure()) return StatusCode::FAILURE;

  m_ths = service("THistSvc", true);

  t_vertices = new TTree("vertices", "vertices ntuple");
  if (m_ths->regTree("/rec/vertices", t_vertices).isFailure()) {
    error() << "Couldn't register vertices tree" << endmsg;
    return StatusCode::FAILURE;
  }
  t_particles = new TTree("particles", "particles ntuple");
  if (m_ths->regTree("/rec/particles", t_particles).isFailure()) {
    error() << "Couldn't register particles tree" << endmsg;
    return StatusCode::FAILURE;
  }
  t_clusters = new TTree("clusters", "clusters ntuple");
  if (m_ths->regTree("/rec/clusters", t_clusters).isFailure()) {
    error() << "Couldn't register clusters tree" << endmsg;
    return StatusCode::FAILURE;
  }
  t_tracks = new TTree("tracks", "tracks ntuple");
  if (m_ths->regTree("/rec/tracks", t_tracks).isFailure()) {
    error() << "Couldn't register tracks tree" << endmsg;
    return StatusCode::FAILURE;
  }

  initializeTrees();
  m_backend = createBackend<1>();
  bool isOk = setupBackend<1>(m_backend, m_dc, m_rhoc, m_dm, m_seed_dc, m_pointsPerBin);
  if (not isOk)
    error() << "No available device";
  auto deviceName = alpaka::getName(alpaka::getDev(backendQueue<1>(m_backend)));
  info() << "CLUEAlgo will run on device " << deviceName << endmsg;

  return StatusCode::SUCCESS;
}

// =====================================================================
// 1) CLUE-based vertexing on (zip, pt)
// =====================================================================
std::vector<int> CLUEVertexing::runClueVertexing(const std::vector<float>& zip,
                                                   const std::vector<float>& pt) const {
  const int n = static_cast<int>(zip.size());
  std::vector<int> clusterIDs(n, -1);
  if (n == 0) return clusterIDs;

  std::vector<float> floatBuffer(n * 2);
  std::vector<int> intBuffer(n * 2);
  std::copy(zip.begin(), zip.end(), floatBuffer.begin());
  std::copy(pt.begin(), pt.end(), floatBuffer.begin() + n);
  clue::PointsHost<1> points(backendQueue<1>(m_backend), n, floatBuffer.data(), intBuffer.data());

  auto clusters = launchVertexing(m_backend, points);

  for (int i = 0; i < n; ++i) {
    clusterIDs[i] = points.clusterIndexes()[i];
    verbose() << "Point #" << i << " : (zip, pt) = (" <<  zip[i] << ","<< pt[i] << ")"
              << " is in cluster " << clusterIDs[i] << endmsg;
  }
  return clusterIDs;
}

// =====================================================================
// PCA direction estimate from cluster hit positions
// =====================================================================
edm4hep::Vector3f CLUEVertexing::estimateDirectionPCA(const std::vector<float>& x,
                                                        const std::vector<float>& y,
                                                        const std::vector<float>& z) const {
  const int n = static_cast<int>(x.size());
  if (n < 2) {
    // Not enough points for PCA — fall back to straight-to-origin direction
    if (n == 1) {
      const float norm = std::sqrt(x[0] * x[0] + y[0] * y[0] + z[0] * z[0]);
      if (norm > 1e-6f) return {x[0] / norm, y[0] / norm, z[0] / norm};
    }
    return {0.f, 0.f, 1.f};
  }

  const float mx = std::accumulate(x.begin(), x.end(), 0.f) / n;
  const float my = std::accumulate(y.begin(), y.end(), 0.f) / n;
  const float mz = std::accumulate(z.begin(), z.end(), 0.f) / n;

  // 3x3 covariance matrix
  float cxx = 0, cxy = 0, cxz = 0, cyy = 0, cyz = 0, czz = 0;
  for (int i = 0; i < n; ++i) {
    const float dx = x[i] - mx, dy = y[i] - my, dz = z[i] - mz;
    cxx += dx * dx; cxy += dx * dy; cxz += dx * dz;
    cyy += dy * dy; cyz += dy * dz; czz += dz * dz;
  }

  // Power iteration for dominant eigenvector (avoids external linalg dependency)
  float vx = 1.f, vy = 1.f, vz = 1.f;
  for (int iter = 0; iter < 50; ++iter) {
    const float nx = cxx * vx + cxy * vy + cxz * vz;
    const float ny = cxy * vx + cyy * vy + cyz * vz;
    const float nz = cxz * vx + cyz * vy + czz * vz;
    const float norm = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (norm < 1e-9f) break;
    vx = nx / norm; vy = ny / norm; vz = nz / norm;
  }

  // Orient outward from the origin, consistent with "away from IP" hypothesis
  const float dot = vx * mx + vy * my + vz * mz;
  if (dot < 0) { vx = -vx; vy = -vy; vz = -vz; }

  return {vx, vy, vz};
}

std::pair<float, float> CLUEVertexing::propagateClusterTime(const edm4hep::Vector3f& clusterPos,
                                                               const edm4hep::Vector3f& direction,
                                                               float beta) const {
  // Straight-line path length from origin along `direction` to the cluster,
  // projected onto that direction (used when no track is available).
  const float pathLength = std::sqrt(clusterPos.x * clusterPos.x +
                                      clusterPos.y * clusterPos.y +
                                      clusterPos.z * clusterPos.z);
  const float dt = pathLength / (beta * C_MM_PER_NS);
  return {pathLength, dt};
}

// =====================================================================
// 2) Fill track / cluster info per particle
// =====================================================================
bool CLUEVertexing::fillTrackInfo(const edm4hep::ReconstructedParticle& part,
                                   const edm4hep::Vector3f& target, TrackInfo& out) const {
  if (part.getTracks().empty()) return false;

  const auto& track = part.getTracks()[0]; // use the first track

  auto ts1 = *std::find_if(track.getTrackStates().begin(), track.getTrackStates().end(),
                            [](auto ts) { return static_cast<int>(ts.location) ==
                                                 static_cast<int>(TrackStateLocation::AtIP); });
  auto ts2 = *std::find_if(track.getTrackStates().begin(), track.getTrackStates().end(),
                            [](auto ts) { return static_cast<int>(ts.location) ==
                                                 static_cast<int>(TrackStateLocation::AtLastHit); });

  out.D0 = ts2.D0; out.phi = ts2.phi; out.omega = ts2.omega;
  out.Z0 = ts2.Z0; out.tanLambda = ts2.tanLambda;
  out.refX = ts2.referencePoint.x; out.refY = ts2.referencePoint.y; out.refZ = ts2.referencePoint.z;
  out.zip = ts1.Z0;

  const auto [pT1, p1] = computeMomentum(ts1);
  const auto [pT, p] = computeMomentum(ts2);
  out.pt = pT1;
  out.beta = p / std::sqrt(p * p + M_PION * M_PION);

  const auto hits = track.getTrackerHits();
  out.time = hits.empty() ? -99.f : hits.back().getTime();

  const auto [path, deltaT] = computeHelixPathAndTimeToPoint(ts2, target, out.beta);
  out.path = path;
  out.t0 = out.time - deltaT;

  return true;
}

std::vector<ClusterInfo> CLUEVertexing::fillClusterInfo(const edm4hep::ReconstructedParticle& part,
                                                          bool hasTrack,
                                                          const TrackInfo& trackAtLastHit) const {
  std::vector<ClusterInfo> result;
  result.reserve(part.getClusters().size());

  for (const auto& cl : part.getClusters()) {
    ClusterInfo ci;
    ci.energy = cl.getEnergy();
    ci.x = cl.getPosition().x; ci.y = cl.getPosition().y; ci.z = cl.getPosition().z;

    float sumT = 0.f;
    for (const auto& hit : cl.getHits()) {
      ci.hitsX.push_back(hit.getPosition().x);
      ci.hitsY.push_back(hit.getPosition().y);
      ci.hitsZ.push_back(hit.getPosition().z);
      ci.hitsTime.push_back(hit.getTime());
      sumT += hit.getTime();
    }
    ci.time = cl.getHits().empty() ? -99.f : sumT / cl.getHits().size();

    if (hasTrack) {
      // Reconstruct the AtLastHit TrackState to propagate to this cluster
      const auto& track = part.getTracks()[0];
      auto ts2 = *std::find_if(track.getTrackStates().begin(), track.getTrackStates().end(),
                                [](auto ts) { return static_cast<int>(ts.location) ==
                                                     static_cast<int>(TrackStateLocation::AtLastHit); });
      const auto [path, deltaT] = computeHelixPathAndTimeToPoint(ts2, cl.getPosition(), trackAtLastHit.beta);
      ci.propTime = deltaT;
      ci.propPath = path;
    } else {
      // No track: infer direction via PCA over this cluster's own hits,
      // or fall back to a straight line from the origin to the cluster position.
      edm4hep::Vector3f direction;
      if (ci.hitsX.size() >= 2) {
        direction = estimateDirectionPCA(ci.hitsX, ci.hitsY, ci.hitsZ);
      } else {
        direction = {0.f, 0.f, 1.f}; // arbitrary, unused by propagateClusterTime
      }
      const auto [path, deltaT] = propagateClusterTime(cl.getPosition(), direction, /*beta=*/1.0f);
      ci.propTime = deltaT;
      ci.propPath = path;
    }

    result.push_back(std::move(ci));
  }
  return result;
}

// =====================================================================
// 3) Vertex time = average of all particle times
// =====================================================================
float CLUEVertexing::computeParticleTime(const ParticleInfo& p) const {
  std::vector<float> times;
  std::vector<float> errors;

  if (p.hasTrack) {
    times.push_back(p.track.t0);
    errors.push_back(0.03f); // track timing resolution, tune as needed
  }
  for (const auto& cl : p.clusters) {
    // cluster time propagated back using propTime computed in fillClusterInfo
    times.push_back(cl.time - cl.propTime);
    errors.push_back(0.1f); // cluster timing resolution, tune as needed
  }

  if (times.empty()) return -99.f;
  const auto [mean, err] = computeWeightedMeanAndStd(times, errors);
  return mean;
}

// =====================================================================
// Main entry point
// =====================================================================
void CLUEVertexing::operator()(const VertexColl& /*vtx_coll*/, const PartColl& part_coll) const {
  cleanTrees();

  // ---- Step 1: build (zip, pt) per particle for CLUE vertexing ----
  std::vector<float> zip_all, pt_all;
  std::vector<int> partIndexForClue;

  for (const auto& part : part_coll) {
    if (part.getTracks().empty()) continue; // CLUE input requires a track-based zip/pt for now
    const auto& track = part.getTracks()[0];
    auto ts1 = *std::find_if(track.getTrackStates().begin(), track.getTrackStates().end(),
                              [](auto ts) { return static_cast<int>(ts.location) ==
                                                   static_cast<int>(TrackStateLocation::AtIP); });
    const auto [pT1, p1] = computeMomentum(ts1);
    zip_all.push_back(ts1.Z0);
    pt_all.push_back(pT1);
    partIndexForClue.push_back(part.id().index);
  }

  const std::vector<int> clusterIDs = runClueVertexing(zip_all, pt_all);

  // Group particle indices by CLUE vertex id
  std::unordered_map<int, std::vector<int>> vertexToParticles;
  for (size_t i = 0; i < clusterIDs.size(); ++i) {
    if (clusterIDs[i] < 0) continue; // outlier / noise, not associated to a vertex
    vertexToParticles[clusterIDs[i]].push_back(partIndexForClue[i]);
  }

  // ---- Step 2 & 3: loop over particles, fill track/cluster info, average vertex time ----
  int globalTrackIndex = 0, globalClusterIndex = 0, globalParticleIndex = 0;

  for (const auto& [vtxID, partIndices] : vertexToParticles) {
    std::vector<float> vertexTimes;

    for (const auto& part : part_coll) {
      if (std::find(partIndices.begin(), partIndices.end(), part.id().index) == partIndices.end())
        continue;

      ParticleInfo pinfo;
      pinfo.hasTrack = !part.getTracks().empty();
      pinfo.usedInVertex = true;

      // target for track propagation: use vertex position once available,
      // otherwise fall back to origin (placeholder until vtx_coll is filled in)
      const edm4hep::Vector3f target{0.f, 0.f, 0.f};

      std::vector<int> p_tracks, p_clusters;

      if (pinfo.hasTrack) {
        fillTrackInfo(part, target, pinfo.track);

        trk_D0.push_back(pinfo.track.D0);
        trk_phi.push_back(pinfo.track.phi);
        trk_omega.push_back(pinfo.track.omega);
        trk_Z0.push_back(pinfo.track.Z0);
        trk_tanLambda.push_back(pinfo.track.tanLambda);
        trk_refPointX.push_back(pinfo.track.refX);
        trk_refPointY.push_back(pinfo.track.refY);
        trk_refPointZ.push_back(pinfo.track.refZ);
        trk_zip.push_back(pinfo.track.zip);
        trk_pt.push_back(pinfo.track.pt);
        track_time.push_back(pinfo.track.time);
        track_t0.push_back(pinfo.track.t0);
        track_path.push_back(pinfo.track.path);
        track_beta.push_back(pinfo.track.beta);
        p_tracks.push_back(globalTrackIndex++);
      }

      pinfo.clusters = fillClusterInfo(part, pinfo.hasTrack, pinfo.track);
      for (const auto& ci : pinfo.clusters) {
        clus_energy.push_back(ci.energy);
        clus_x.push_back(ci.x); clus_y.push_back(ci.y); clus_z.push_back(ci.z);
        clus_time.push_back(ci.time);
        clus_propTime.push_back(ci.propTime);
        clus_propPath.push_back(ci.propPath);
        clus_hits_x.push_back(ci.hitsX);
        clus_hits_y.push_back(ci.hitsY);
        clus_hits_z.push_back(ci.hitsZ);
        clus_hits_time.push_back(ci.hitsTime);
        p_clusters.push_back(globalClusterIndex++);
      }

      pinfo.particleTime = computeParticleTime(pinfo);
      vertexTimes.push_back(pinfo.particleTime);

      part_tracks.push_back(p_tracks);
      part_clusters.push_back(p_clusters);
      p_isVertexAssociated.push_back(true);
      p_time.push_back(pinfo.particleTime);
      globalParticleIndex++;
    }

    // ---- vertex time = average over particle times ----
    std::vector<float> validTimes;
    for (float t : vertexTimes) if (t > -90.f) validTimes.push_back(t);

    float vtxTime = -99.f, vtxTimeErr = -1.f;
    if (!validTimes.empty()) {
      std::vector<float> errors(validTimes.size(), 0.05f);
      const auto [mean, err] = computeWeightedMeanAndStd(validTimes, errors);
      vtxTime = mean;
      vtxTimeErr = err;
    }

    v_time.push_back(vtxTime);
    v_timeErr.push_back(vtxTimeErr);
    // v_x, v_y, v_z can be filled once CLUE also gives a spatial vertex position
  }

  t_vertices->Fill();
  t_particles->Fill();
  t_tracks->Fill();
  t_clusters->Fill();
}

void CLUEVertexing::initializeTrees() {
  t_vertices->Branch("x", &v_x);
  t_vertices->Branch("y", &v_y);
  t_vertices->Branch("z", &v_z);
  t_vertices->Branch("time", &v_time);
  t_vertices->Branch("timeErr", &v_timeErr);
  t_vertices->Branch("particles", &v_particles);

  t_particles->Branch("tracks", &part_tracks);
  t_particles->Branch("clusters", &part_clusters);
  t_particles->Branch("hasVertex", &p_isVertexAssociated);
  t_particles->Branch("time", &p_time);

  t_tracks->Branch("time", &track_time);
  t_tracks->Branch("t0", &track_t0);
  t_tracks->Branch("path", &track_path);
  t_tracks->Branch("beta", &track_beta);
  t_tracks->Branch("trk_D0", &trk_D0);
  t_tracks->Branch("trk_phi", &trk_phi);
  t_tracks->Branch("trk_omega", &trk_omega);
  t_tracks->Branch("trk_Z0", &trk_Z0);
  t_tracks->Branch("trk_tanLambda", &trk_tanLambda);
  t_tracks->Branch("trk_refPointX", &trk_refPointX);
  t_tracks->Branch("trk_refPointY", &trk_refPointY);
  t_tracks->Branch("trk_refPointZ", &trk_refPointZ);
  t_tracks->Branch("zip", &trk_zip);
  t_tracks->Branch("pt", &trk_pt);

  t_clusters->Branch("energy", &clus_energy);
  t_clusters->Branch("x", &clus_x);
  t_clusters->Branch("y", &clus_y);
  t_clusters->Branch("z", &clus_z);
  t_clusters->Branch("time", &clus_time);
  t_clusters->Branch("propTime", &clus_propTime);
  t_clusters->Branch("propPath", &clus_propPath);
  t_clusters->Branch("hits_x", &clus_hits_x);
  t_clusters->Branch("hits_y", &clus_hits_y);
  t_clusters->Branch("hits_z", &clus_hits_z);
  t_clusters->Branch("hits_time", &clus_hits_time);
}

void CLUEVertexing::cleanTrees() const {
  v_x.clear();
  v_y.clear();
  v_z.clear();
  v_time.clear();
  v_timeErr.clear();
  v_particles.clear();
  part_tracks.clear();
  part_clusters.clear();
  p_isVertexAssociated.clear();
  p_time.clear();
  track_time.clear();
  track_t0.clear();
  track_path.clear();
  track_beta.clear();
  trk_D0.clear();
  trk_phi.clear();
  trk_omega.clear();
  trk_Z0.clear();
  trk_tanLambda.clear();
  trk_refPointX.clear();
  trk_refPointY.clear();
  trk_refPointZ.clear();
  trk_zip.clear();
  trk_pt.clear();
  clus_energy.clear();
  clus_x.clear();
  clus_y.clear();
  clus_z.clear();
  clus_time.clear();
  clus_propTime.clear();
  clus_propPath.clear();
  clus_hits_x.clear();
  clus_hits_y.clear();
  clus_hits_z.clear();
  clus_hits_time.clear();
}

StatusCode CLUEVertexing::finalize() {
  if (Gaudi::Algorithm::finalize().isFailure()) return StatusCode::FAILURE;
  return StatusCode::SUCCESS;
}
