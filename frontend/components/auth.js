(function () {
    'use strict';

    const USERS_KEY = 'aquamind.auth.users.v1';
    const SESSION_KEY = 'aquamind.auth.session.v1';
    const LOGIN_PAGE = 'login.html';
    const SESSION_MAX_AGE = 30 * 24 * 60 * 60 * 1000;

    function readJSON(storage, key, fallback) {
        try {
            const value = JSON.parse(storage.getItem(key) || 'null');
            return value === null ? fallback : value;
        } catch (_) {
            return fallback;
        }
    }

    function getUsers() {
        const users = readJSON(localStorage, USERS_KEY, []);
        return Array.isArray(users) ? users : [];
    }

    function normalizeUsername(username) {
        return String(username || '').trim().toLocaleLowerCase('zh-CN');
    }

    function randomSalt() {
        const bytes = new Uint8Array(16);
        if (window.crypto && window.crypto.getRandomValues) {
            window.crypto.getRandomValues(bytes);
        } else {
            for (let i = 0; i < bytes.length; i += 1) bytes[i] = Math.floor(Math.random() * 256);
        }
        return Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
    }

    function fallbackHash(value) {
        let h1 = 0x811c9dc5;
        let h2 = 0x9e3779b9;
        for (let i = 0; i < value.length; i += 1) {
            h1 = Math.imul(h1 ^ value.charCodeAt(i), 0x01000193);
            h2 = Math.imul(h2 ^ value.charCodeAt(i), 0x85ebca6b);
        }
        return `${(h1 >>> 0).toString(16).padStart(8, '0')}${(h2 >>> 0).toString(16).padStart(8, '0')}`;
    }

    async function hashPassword(password, salt) {
        const source = `${salt}:${String(password)}`;
        if (window.crypto && window.crypto.subtle && window.TextEncoder) {
            const digest = await window.crypto.subtle.digest('SHA-256', new TextEncoder().encode(source));
            return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
        }
        return fallbackHash(source);
    }

    function readSession() {
        const candidates = [
            readJSON(sessionStorage, SESSION_KEY, null),
            readJSON(localStorage, SESSION_KEY, null)
        ];
        const session = candidates.find(item => item && item.username && item.createdAt);
        if (!session) return null;
        const accountExists = getUsers().some(user => user.normalized === normalizeUsername(session.username));
        if (!accountExists || Date.now() - Number(session.createdAt) > SESSION_MAX_AGE) {
            sessionStorage.removeItem(SESSION_KEY);
            localStorage.removeItem(SESSION_KEY);
            return null;
        }
        return session;
    }

    function writeSession(username, remember) {
        const session = { username, createdAt: Date.now() };
        sessionStorage.removeItem(SESSION_KEY);
        localStorage.removeItem(SESSION_KEY);
        (remember ? localStorage : sessionStorage).setItem(SESSION_KEY, JSON.stringify(session));
        return session;
    }

    async function register(username, password, remember) {
        const displayName = String(username || '').trim();
        const normalized = normalizeUsername(displayName);
        if (displayName.length < 2 || displayName.length > 20) {
            throw new Error('用户名请输入 2–20 个字符');
        }
        if (!/^[\u4e00-\u9fa5A-Za-z0-9_.-]+$/.test(displayName)) {
            throw new Error('用户名只能包含中文、字母、数字、下划线、点或短横线');
        }
        if (String(password || '').length < 6 || String(password || '').length > 64) {
            throw new Error('密码请输入 6–64 个字符');
        }

        const users = getUsers();
        if (users.some(user => user.normalized === normalized)) {
            throw new Error('该用户名已注册，请直接登录');
        }
        const salt = randomSalt();
        const passwordHash = await hashPassword(password, salt);
        users.push({
            username: displayName,
            normalized,
            salt,
            passwordHash,
            createdAt: new Date().toISOString()
        });
        localStorage.setItem(USERS_KEY, JSON.stringify(users));
        writeSession(displayName, remember);
        return { username: displayName };
    }

    async function login(username, password, remember) {
        const normalized = normalizeUsername(username);
        const user = getUsers().find(item => item.normalized === normalized);
        if (!user) throw new Error('用户名或密码错误');
        const passwordHash = await hashPassword(password, user.salt);
        if (passwordHash !== user.passwordHash) throw new Error('用户名或密码错误');
        writeSession(user.username, remember);
        return { username: user.username };
    }

    function logout() {
        sessionStorage.removeItem(SESSION_KEY);
        localStorage.removeItem(SESSION_KEY);
        window.location.replace(LOGIN_PAGE);
    }

    function safeNextPage(value) {
        const allowed = new Set([
            'smart_water_system.html', 'process_map.html', 'smart_water_system_v2.html',
            'ai_dosage.html', 'device_health.html', 'energy_schedule.html', 'data_insight.html'
        ]);
        return allowed.has(value) ? value : 'smart_water_system.html';
    }

    function mountAccountControl() {
        const session = readSession();
        if (!session || document.getElementById('authAccountControl')) return;
        const target = document.querySelector('.page-actions');
        if (!target) return;

        if (!document.getElementById('authAccountStyles')) {
            const style = document.createElement('style');
            style.id = 'authAccountStyles';
            style.textContent = `
                .auth-account { position: relative; display: flex; align-items: center; }
                .auth-account__trigger { height: 36px; display: inline-flex; align-items: center; gap: 8px; padding: 0 12px; border: 1px solid var(--border-color, #dce6eb); border-radius: 9px; background: #fff; color: var(--text-primary, #172b36); cursor: pointer; font: inherit; }
                .auth-account__trigger:hover { border-color: var(--primary, #0793a6); background: #f2fbfc; }
                .auth-account__avatar { width: 24px; height: 24px; display: grid; place-items: center; border-radius: 50%; background: linear-gradient(135deg, #0b91a4, #18ad83); color: #fff; font-size: 11px; font-weight: 700; }
                .auth-account__menu { position: absolute; z-index: 1000; top: calc(100% + 8px); right: 0; min-width: 150px; padding: 6px; border: 1px solid #dce6eb; border-radius: 10px; background: #fff; box-shadow: 0 12px 30px rgba(28, 61, 75, .14); display: none; }
                .auth-account.open .auth-account__menu { display: block; }
                .auth-account__menu button { width: 100%; border: 0; border-radius: 7px; background: transparent; padding: 9px 10px; text-align: left; color: #4b6170; cursor: pointer; font: inherit; }
                .auth-account__menu button:hover { color: #d44747; background: #fff3f3; }
                @media (max-width: 640px) { .auth-account__name { display: none; } }
            `;
            document.head.appendChild(style);
        }

        const wrapper = document.createElement('div');
        wrapper.id = 'authAccountControl';
        wrapper.className = 'auth-account';
        const trigger = document.createElement('button');
        trigger.type = 'button';
        trigger.className = 'auth-account__trigger';
        trigger.setAttribute('aria-haspopup', 'true');
        trigger.setAttribute('aria-expanded', 'false');

        const avatar = document.createElement('span');
        avatar.className = 'auth-account__avatar';
        avatar.textContent = session.username.slice(0, 1).toUpperCase();
        const name = document.createElement('span');
        name.className = 'auth-account__name';
        name.textContent = session.username;
        const chevron = document.createElement('i');
        chevron.className = 'fas fa-chevron-down';
        chevron.style.fontSize = '10px';
        trigger.append(avatar, name, chevron);

        const menu = document.createElement('div');
        menu.className = 'auth-account__menu';
        const logoutButton = document.createElement('button');
        logoutButton.type = 'button';
        logoutButton.innerHTML = '<i class="fas fa-right-from-bracket" style="margin-right:8px"></i>退出登录';
        logoutButton.addEventListener('click', logout);
        menu.appendChild(logoutButton);
        wrapper.append(trigger, menu);
        target.appendChild(wrapper);

        trigger.addEventListener('click', event => {
            event.stopPropagation();
            const open = wrapper.classList.toggle('open');
            trigger.setAttribute('aria-expanded', String(open));
        });
        document.addEventListener('click', () => {
            wrapper.classList.remove('open');
            trigger.setAttribute('aria-expanded', 'false');
        });
    }

    const currentPage = (window.location.pathname.split('/').pop() || '').toLowerCase();
    const isLoginPage = currentPage === LOGIN_PAGE;
    const activeSession = readSession();

    window.AquaMindAuth = {
        register,
        login,
        logout,
        getSession: readSession,
        hasUsers: () => getUsers().length > 0,
        safeNextPage
    };

    if (!isLoginPage && !activeSession) {
        document.documentElement.style.visibility = 'hidden';
        const next = safeNextPage(currentPage);
        window.location.replace(`${LOGIN_PAGE}?next=${encodeURIComponent(next)}`);
        return;
    }

    if (!isLoginPage) {
        document.addEventListener('DOMContentLoaded', () => window.setTimeout(mountAccountControl, 0));
    }
})();
