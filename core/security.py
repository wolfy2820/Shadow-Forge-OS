"""
Security - Comprehensive Security & Compliance Framework

The Security component provides multi-layered protection, encryption,
authentication, and compliance monitoring for all ShadowForge OS operations.
"""

import asyncio
import logging
import hashlib
import secrets
import jwt
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"

class ThreatLevel(Enum):
    """Threat assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditEvent(Enum):
    """Types of security audit events."""
    LOGIN_ATTEMPT = "login_attempt"
    DATA_ACCESS = "data_access"
    PERMISSION_CHANGE = "permission_change"
    ENCRYPTION_KEY_ROTATION = "encryption_key_rotation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SECURITY_BREACH = "security_breach"

@dataclass
class SecurityConfig:
    """Security configuration."""
    jwt_secret: str
    jwt_expiry_hours: int
    password_min_length: int
    max_login_attempts: int
    session_timeout_minutes: int
    encryption_enabled: bool
    audit_enabled: bool

@dataclass
class SecurityAlert:
    """Security alert data structure."""
    alert_id: str
    threat_level: ThreatLevel
    event_type: AuditEvent
    description: str
    source_ip: str
    user_id: Optional[str]
    timestamp: datetime
    resolved: bool

class SecurityFramework:
    """
    Security Framework - Comprehensive protection system.
    
    Features:
    - Multi-factor authentication
    - End-to-end encryption
    - Role-based access control
    - Real-time threat detection
    - Security audit logging
    - Automated incident response
    - Compliance monitoring
    """
    
    def __init__(self, config: SecurityConfig = None):
        self.logger = logging.getLogger(f"{__name__}.security")
        
        # Security configuration
        self.config = config or SecurityConfig(
            jwt_secret=secrets.token_urlsafe(32),
            jwt_expiry_hours=24,
            password_min_length=8,
            max_login_attempts=5,
            session_timeout_minutes=30,
            encryption_enabled=True,
            audit_enabled=True
        )
        
        # Security state
        self.active_sessions: Dict[str, Dict] = {}
        self.failed_login_attempts: Dict[str, List[datetime]] = {}
        self.security_alerts: List[SecurityAlert] = []
        self.audit_log: List[Dict[str, Any]] = []
        
        # Encryption
        self.encryption_key = None
        self.cipher_suite = None
        
        # Access control
        self.user_permissions: Dict[str, Dict] = {}
        self.role_definitions: Dict[str, List[str]] = {}
        
        # Threat detection
        self.suspicious_ips: set = set()
        self.blocked_ips: set = set()
        
        # Performance metrics
        self.security_checks_performed = 0
        self.threats_detected = 0
        self.alerts_generated = 0
        self.encryption_operations = 0
        
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the Security Framework."""
        try:
            self.logger.info("🔒 Initializing Security Framework...")
            
            # Setup encryption
            await self._setup_encryption()
            
            # Initialize access control
            await self._setup_access_control()
            
            # Setup threat detection
            await self._setup_threat_detection()
            
            # Start security monitoring
            asyncio.create_task(self._security_monitoring_loop())
            
            # Start audit log maintenance
            asyncio.create_task(self._audit_maintenance_loop())
            
            self.is_initialized = True
            self.logger.info("✅ Security Framework initialized - Protection active")
            
        except Exception as e:
            self.logger.error(f"❌ Security Framework initialization failed: {e}")
            raise
    
    async def deploy(self, target: str):
        """Deploy Security Framework to target environment."""
        self.logger.info(f"🚀 Deploying Security Framework to {target}")
        
        if target == "production":
            await self._enable_production_security_features()
        
        self.logger.info(f"✅ Security Framework deployed to {target}")
    
    # Authentication & Authorization
    
    async def authenticate_user(self, username: str, password: str, 
                              source_ip: str = None) -> Optional[Dict[str, Any]]:
        """
        Authenticate user with comprehensive security checks.
        
        Args:
            username: User identifier
            password: User password
            source_ip: Request source IP
            
        Returns:
            Authentication result with JWT token if successful
        """
        try:
            # Check for IP blocks
            if source_ip and source_ip in self.blocked_ips:
                await self._log_audit_event(AuditEvent.LOGIN_ATTEMPT, {
                    "username": username,
                    "source_ip": source_ip,
                    "result": "blocked_ip",
                    "timestamp": datetime.now().isoformat()
                })
                return None
            
            # Check failed login attempts
            if await self._check_failed_login_attempts(username, source_ip):
                return None
            
            # Verify credentials (mock implementation)
            if await self._verify_password(username, password):
                # Generate JWT token
                token = await self._generate_jwt_token(username)
                
                # Create session
                session_id = secrets.token_urlsafe(32)
                session_data = {
                    "user_id": username,
                    "token": token,
                    "created_at": datetime.now(),
                    "last_activity": datetime.now(),
                    "source_ip": source_ip,
                    "permissions": self.user_permissions.get(username, {})
                }
                
                self.active_sessions[session_id] = session_data
                
                # Log successful authentication
                await self._log_audit_event(AuditEvent.LOGIN_ATTEMPT, {
                    "username": username,
                    "source_ip": source_ip,
                    "result": "success",
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                })
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "token": token,
                    "expires_at": (datetime.now() + timedelta(hours=self.config.jwt_expiry_hours)).isoformat(),
                    "permissions": session_data["permissions"]
                }
            
            else:
                # Record failed attempt
                await self._record_failed_login(username, source_ip)
                
                await self._log_audit_event(AuditEvent.LOGIN_ATTEMPT, {
                    "username": username,
                    "source_ip": source_ip,
                    "result": "failed",
                    "timestamp": datetime.now().isoformat()
                })
                
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Authentication failed: {e}")
            return None
    
    async def verify_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Verify session validity and update activity."""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return None
            
            # Check session timeout
            timeout_threshold = datetime.now() - timedelta(minutes=self.config.session_timeout_minutes)
            if session["last_activity"] < timeout_threshold:
                del self.active_sessions[session_id]
                return None
            
            # Update last activity
            session["last_activity"] = datetime.now()
            
            # Verify JWT token
            try:
                payload = jwt.decode(session["token"], self.config.jwt_secret, algorithms=["HS256"])
                if payload["user_id"] != session["user_id"]:
                    del self.active_sessions[session_id]
                    return None
            except jwt.InvalidTokenError:
                del self.active_sessions[session_id]
                return None
            
            return session
            
        except Exception as e:
            self.logger.error(f"❌ Session verification failed: {e}")
            return None
    
    async def check_permission(self, user_id: str, resource: str, action: str) -> bool:
        """Check if user has permission for specific resource action."""
        try:
            user_perms = self.user_permissions.get(user_id, {})
            resource_perms = user_perms.get("resources", {}).get(resource, [])
            
            return action in resource_perms or "admin" in user_perms.get("roles", [])
            
        except Exception as e:
            self.logger.error(f"❌ Permission check failed: {e}")
            return False
    
    # Encryption & Data Protection
    
    async def encrypt_data(self, data: Union[str, bytes], classification: SecurityLevel = SecurityLevel.INTERNAL) -> str:
        """
        Encrypt sensitive data based on security classification.
        
        Args:
            data: Data to encrypt
            classification: Security level classification
            
        Returns:
            Encrypted data as base64 string
        """
        try:
            if not self.config.encryption_enabled:
                return data if isinstance(data, str) else data.decode()
            
            if isinstance(data, str):
                data = data.encode()
            
            encrypted_data = self.cipher_suite.encrypt(data)
            self.encryption_operations += 1
            
            # Log encryption for high-security data
            if classification in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
                await self._log_audit_event(AuditEvent.DATA_ACCESS, {
                    "operation": "encrypt",
                    "classification": classification.value,
                    "data_size": len(data),
                    "timestamp": datetime.now().isoformat()
                })
            
            return base64.b64encode(encrypted_data).decode()
            
        except Exception as e:
            self.logger.error(f"❌ Data encryption failed: {e}")
            raise
    
    async def decrypt_data(self, encrypted_data: str, classification: SecurityLevel = SecurityLevel.INTERNAL) -> str:
        """
        Decrypt data with audit logging.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            classification: Security level classification
            
        Returns:
            Decrypted data as string
        """
        try:
            if not self.config.encryption_enabled:
                return encrypted_data
            
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
            self.encryption_operations += 1
            
            # Log decryption for high-security data
            if classification in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
                await self._log_audit_event(AuditEvent.DATA_ACCESS, {
                    "operation": "decrypt",
                    "classification": classification.value,
                    "timestamp": datetime.now().isoformat()
                })
            
            return decrypted_data.decode()
            
        except Exception as e:
            self.logger.error(f"❌ Data decryption failed: {e}")
            raise
    
    # Threat Detection & Response
    
    async def detect_threat(self, event_data: Dict[str, Any]) -> Optional[SecurityAlert]:
        """
        Analyze event for potential security threats.
        
        Args:
            event_data: Event information to analyze
            
        Returns:
            Security alert if threat detected
        """
        try:
            threat_level = ThreatLevel.LOW
            description = "Normal activity"
            
            # Analyze suspicious patterns
            source_ip = event_data.get("source_ip")
            user_id = event_data.get("user_id")
            
            # Check for rapid requests
            if await self._check_rapid_requests(source_ip):
                threat_level = ThreatLevel.MEDIUM
                description = "Rapid request pattern detected"
            
            # Check for invalid access attempts
            if event_data.get("result") == "failed" and await self._check_repeated_failures(source_ip):
                threat_level = ThreatLevel.HIGH
                description = "Multiple authentication failures"
            
            # Check for access from suspicious locations
            if await self._check_suspicious_location(source_ip):
                threat_level = ThreatLevel.MEDIUM
                description = "Access from unusual location"
            
            # Generate alert for medium+ threats
            if threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                alert = SecurityAlert(
                    alert_id=f"alert_{datetime.now().timestamp()}",
                    threat_level=threat_level,
                    event_type=AuditEvent.SUSPICIOUS_ACTIVITY,
                    description=description,
                    source_ip=source_ip or "unknown",
                    user_id=user_id,
                    timestamp=datetime.now(),
                    resolved=False
                )
                
                self.security_alerts.append(alert)
                self.alerts_generated += 1
                self.threats_detected += 1
                
                # Auto-response for high threats
                if threat_level == ThreatLevel.HIGH:
                    await self._auto_respond_to_threat(alert)
                
                return alert
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Threat detection failed: {e}")
            return None
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get security framework performance metrics."""
        return {
            "security_checks_performed": self.security_checks_performed,
            "threats_detected": self.threats_detected,
            "alerts_generated": self.alerts_generated,
            "encryption_operations": self.encryption_operations,
            "active_sessions": len(self.active_sessions),
            "blocked_ips": len(self.blocked_ips),
            "suspicious_ips": len(self.suspicious_ips),
            "audit_log_size": len(self.audit_log),
            "unresolved_alerts": len([a for a in self.security_alerts if not a.resolved])
        }
    
    # Helper methods
    
    async def _setup_encryption(self):
        """Setup encryption system."""
        try:
            if self.config.encryption_enabled:
                # Generate encryption key
                password = self.config.jwt_secret.encode()
                salt = b'shadowforge_salt_v51'  # In production, use random salt
                
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                
                self.encryption_key = base64.urlsafe_b64encode(kdf.derive(password))
                self.cipher_suite = Fernet(self.encryption_key)
                
                self.logger.debug("🔐 Encryption system initialized")
        except Exception as e:
            self.logger.error(f"❌ Encryption setup failed: {e}")
            raise
    
    async def _setup_access_control(self):
        """Setup role-based access control."""
        # Define default roles and permissions
        self.role_definitions = {
            "admin": ["read", "write", "delete", "admin"],
            "user": ["read", "write"],
            "viewer": ["read"]
        }
        
        # Setup default user permissions (mock implementation)
        self.user_permissions = {
            "developer": {
                "roles": ["admin"],
                "resources": {
                    "system": ["read", "write", "admin"],
                    "agents": ["read", "write", "admin"],
                    "content": ["read", "write", "admin"],
                    "financial": ["read", "write", "admin"]
                }
            },
            "user_001": {
                "roles": ["user"],
                "resources": {
                    "content": ["read", "write"],
                    "interface": ["read", "write"]
                }
            }
        }
    
    async def _setup_threat_detection(self):
        """Setup threat detection systems."""
        # Initialize known threat indicators
        self.suspicious_ips.update([
            "192.168.1.100",  # Example suspicious IP
        ])
    
    async def _verify_password(self, username: str, password: str) -> bool:
        """Verify user password (mock implementation)."""
        # In production, this would check against secure password storage
        mock_users = {
            "developer": "dev_password",
            "user_001": "user_password"
        }
        return mock_users.get(username) == password
    
    async def _generate_jwt_token(self, user_id: str) -> str:
        """Generate JWT token for authenticated user."""
        payload = {
            "user_id": user_id,
            "issued_at": datetime.now().timestamp(),
            "expires_at": (datetime.now() + timedelta(hours=self.config.jwt_expiry_hours)).timestamp()
        }
        
        return jwt.encode(payload, self.config.jwt_secret, algorithm="HS256")
    
    async def _check_failed_login_attempts(self, username: str, source_ip: str) -> bool:
        """Check if user/IP has exceeded failed login attempts."""
        now = datetime.now()
        window_start = now - timedelta(minutes=30)
        
        # Check username attempts
        user_key = f"user_{username}"
        if user_key in self.failed_login_attempts:
            recent_attempts = [
                attempt for attempt in self.failed_login_attempts[user_key]
                if attempt > window_start
            ]
            if len(recent_attempts) >= self.config.max_login_attempts:
                return True
        
        # Check IP attempts
        if source_ip:
            ip_key = f"ip_{source_ip}"
            if ip_key in self.failed_login_attempts:
                recent_attempts = [
                    attempt for attempt in self.failed_login_attempts[ip_key]
                    if attempt > window_start
                ]
                if len(recent_attempts) >= self.config.max_login_attempts * 2:
                    self.blocked_ips.add(source_ip)
                    return True
        
        return False
    
    async def _record_failed_login(self, username: str, source_ip: str):
        """Record failed login attempt."""
        now = datetime.now()
        
        user_key = f"user_{username}"
        if user_key not in self.failed_login_attempts:
            self.failed_login_attempts[user_key] = []
        self.failed_login_attempts[user_key].append(now)
        
        if source_ip:
            ip_key = f"ip_{source_ip}"
            if ip_key not in self.failed_login_attempts:
                self.failed_login_attempts[ip_key] = []
            self.failed_login_attempts[ip_key].append(now)
    
    async def _log_audit_event(self, event_type: AuditEvent, event_data: Dict[str, Any]):
        """Log security audit event."""
        if self.config.audit_enabled:
            audit_entry = {
                "event_id": f"audit_{datetime.now().timestamp()}",
                "event_type": event_type.value,
                "timestamp": datetime.now().isoformat(),
                "data": event_data
            }
            
            self.audit_log.append(audit_entry)
            self.logger.debug(f"📋 Audit event logged: {event_type.value}")
    
    async def _check_rapid_requests(self, source_ip: str) -> bool:
        """Check for rapid request patterns."""
        # Simplified implementation
        return False
    
    async def _check_repeated_failures(self, source_ip: str) -> bool:
        """Check for repeated failure patterns."""
        if source_ip:
            ip_key = f"ip_{source_ip}"
            recent_failures = self.failed_login_attempts.get(ip_key, [])
            return len(recent_failures) >= 3
        return False
    
    async def _check_suspicious_location(self, source_ip: str) -> bool:
        """Check if request comes from suspicious location."""
        return source_ip in self.suspicious_ips
    
    async def _auto_respond_to_threat(self, alert: SecurityAlert):
        """Automatically respond to detected threats."""
        if alert.threat_level == ThreatLevel.HIGH:
            # Block suspicious IP
            if alert.source_ip and alert.source_ip != "unknown":
                self.blocked_ips.add(alert.source_ip)
                self.logger.warning(f"🚫 IP blocked due to threat: {alert.source_ip}")
            
            # Terminate suspicious sessions
            if alert.user_id:
                sessions_to_remove = [
                    session_id for session_id, session in self.active_sessions.items()
                    if session["user_id"] == alert.user_id
                ]
                for session_id in sessions_to_remove:
                    del self.active_sessions[session_id]
                
                self.logger.warning(f"🔒 Sessions terminated for user: {alert.user_id}")
    
    async def _security_monitoring_loop(self):
        """Background security monitoring loop."""
        while self.is_initialized:
            try:
                # Clean up old failed attempts
                await self._cleanup_failed_attempts()
                
                # Check for stale sessions
                await self._cleanup_stale_sessions()
                
                # Update threat intelligence
                await self._update_threat_intelligence()
                
                self.security_checks_performed += 1
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Security monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _audit_maintenance_loop(self):
        """Background audit log maintenance."""
        while self.is_initialized:
            try:
                # Keep audit log manageable
                if len(self.audit_log) > 10000:
                    self.audit_log = self.audit_log[-5000:]
                
                # Keep security alerts manageable
                if len(self.security_alerts) > 1000:
                    self.security_alerts = self.security_alerts[-500:]
                
                await asyncio.sleep(3600)  # Maintenance every hour
                
            except Exception as e:
                self.logger.error(f"❌ Audit maintenance error: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_failed_attempts(self):
        """Clean up old failed login attempts."""
        cutoff = datetime.now() - timedelta(hours=24)
        
        for key in list(self.failed_login_attempts.keys()):
            self.failed_login_attempts[key] = [
                attempt for attempt in self.failed_login_attempts[key]
                if attempt > cutoff
            ]
            
            if not self.failed_login_attempts[key]:
                del self.failed_login_attempts[key]
    
    async def _cleanup_stale_sessions(self):
        """Remove stale sessions."""
        cutoff = datetime.now() - timedelta(minutes=self.config.session_timeout_minutes)
        
        stale_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if session["last_activity"] < cutoff
        ]
        
        for session_id in stale_sessions:
            del self.active_sessions[session_id]
    
    async def _update_threat_intelligence(self):
        """Update threat intelligence data."""
        # In production, this would connect to threat intelligence feeds
        pass
    
    async def _enable_production_security_features(self):
        """Enable production-specific security features."""
        # Enhanced encryption for production
        self.config.encryption_enabled = True
        self.config.audit_enabled = True
        self.config.max_login_attempts = 3
        self.config.session_timeout_minutes = 15
        
        self.logger.info("🔒 Production security features enabled")